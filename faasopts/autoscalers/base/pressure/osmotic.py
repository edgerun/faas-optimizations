import datetime
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict

import pandas as pd
from dataclasses_json import dataclass_json
from faas.context import PlatformContext, FunctionReplicaFactory
from faas.system import FunctionReplica, FunctionDeployment
from faas.util.constant import pod_type_label, api_gateway_type_label, zone_label, function_label, pod_pending, \
    worker_role_label

from faasopts.autoscalers.api import BaseAutoscaler
from faasopts.utils.pressure.calculation import LogisticFunctionParameters
from faasopts.utils.pressure.service import PressureService

logger = logging.getLogger(__name__)


@dataclass
class OsmoticScalerParameters:
    violates_threshold: Callable[[PlatformContext, FunctionReplica], bool]
    violates_min_threshold: Callable[[PlatformContext, FunctionReplica], bool]
    # this thresholds are used to determine functions under pressure
    max_threshold: float
    min_threshold: float
    function_requirements: Dict[str, float]
    max_latency: float
    lookback: int
    pressures: List[str]
    max_containers: int
    # either latency (in ms) or rtt (in s)
    target_time_measure: str
    percentile_duration: float = 90
    deployment_pattern: str = '-deployment'
    logistic_function_parameters: Optional[LogisticFunctionParameters] = None


@dataclass_json
@dataclass
class ScaleScheduleEvent:
    ts: float
    fn: str
    replica: FunctionReplica
    origin_zone: str
    delete: bool = False



def get_average_requests_over_replicas(fn: str, traces: pd.DataFrame):
    """
    Calculates the average number of requests over all replicas of one function.
    :param fn: function name
    :return:
    """
    df = traces[traces['function'] == fn]
    return df.groupby('container_id').count()['ts'].mean().mean()





class OsmoticAutoscaler(BaseAutoscaler):

    def __init__(self, parameters: OsmoticScalerParameters, gateway: FunctionReplica,
                 replica_factory: FunctionReplicaFactory, now: Callable[[], float], pressure_service: PressureService):
        self.parameters = parameters
        self.gateway = gateway
        self.cluster = self.gateway.labels[zone_label]
        self.replica_factory = replica_factory
        self.now = now
        self.pressure_service = pressure_service

    def run(self, ctx: PlatformContext) -> List[ScaleScheduleEvent]:
        # TODO probably a generator
        logger.info("start to figure scale up out")
        result = OsmoticScalerSchedulerResult([], [], [], [], [], [], [], [], [], [])
        pressure_values = self.calculate_pressure_per_fn_per_zone(ctx, result)
        if pressure_values is None:
            logger.info("No pressure values calculated, no further execution")
            return result
        pressure_id = self.pressure_service.save(pressure_values)
        delete_results = self.get_scale_down_actions(pressure_values, ctx)
        delete_results_id = self.pressure_service.save_delete_results(delete_results)
        # TODO create dictionary by function based on delete to support multi-function experiments
        create_results = self.get_scale_up_actions(pressure_values, ctx, len(delete_results),
                                                   self.parameters.max_containers, pressure_id, delete_results_id)
        logger.info(f"figured out scaling, {len(create_results)} up scale events")
        logger.info("figure our scale down")
        logger.info(f"figured out down scaling, {len(delete_results)} down scale events")
        all_results = []
        all_results.extend(delete_results)
        all_results.extend(create_results)
        return all_results

    def get_scale_up_actions(self, pressure_values: pd.DataFrame, ctx: PlatformContext, teardowns: int,
                             max_replica: int, pressure_id: str, delete_results_id: str) -> List[ScaleScheduleEvent]:
        """
        Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
        """
        under_pressures = self.is_under_pressure(pressure_values, ctx)
        logger.info(" under pressure_gateway %d", len(under_pressures))
        under_pressure_new = []
        new_pods = {}
        for under_pressure in under_pressures:
            deployment = under_pressure[1]
            running_pods = len(ctx.replica_service.get_function_replicas_of_deployment(deployment.name))
            pending_pods = len(ctx.replica_service.get_function_replicas_of_deployment(deployment.name, running=False,
                                                                                       state=pod_pending))
            all_pods = running_pods + pending_pods
            no_new_pods = new_pods.get(deployment.name, 0)
            if ((all_pods + no_new_pods) - teardowns) < max_replica:
                under_pressure_new.append(under_pressure)
                if new_pods.get(deployment.name, None) is None:
                    new_pods[deployment.name] = 1
                else:
                    new_pods[deployment.name] += 1

        scale_schedule_events = []
        for event in under_pressure_new:
            target_gateway = event[0]
            deployment = event[1]
            violation_origin_gateway = event[2]
            scale_schedule_event = self.prepare_pod_request(deployment, origin_gateway=violation_origin_gateway,
                                                            target_gateway=target_gateway,
                                                            pressure_id=pressure_id,
                                                            delete_results_id=delete_results_id)
            scale_schedule_events.append(scale_schedule_event)
        return scale_schedule_events

    def scheduling_policy(
            self,
            scale_functions: List[Tuple[FunctionReplica, FunctionDeployment, FunctionReplica]], pressure_id: str
    ):
        """
        Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
        """
        for event in scale_functions:
            target_gateway = event[0]
            deployment = event[1]
            violation_origin_gateway = event[2]
            self.prepare_pod_request(deployment, origin_gateway=violation_origin_gateway, target_gateway=target_gateway,
                                     pressure_id=pressure_id)

    def prepare_pod_request(self, deployment: FunctionDeployment, origin_gateway: FunctionReplica,
                            target_gateway: FunctionReplica, pressure_id: str,
                            delete_results_id: str) -> ScaleScheduleEvent:

        replica = self.replica_factory.create_replica(
            {worker_role_label: 'true', 'origin_zone': self.cluster, zone_label: 'None',
             'schedulerName': 'global-scheduler', 'pressure-id': pressure_id, 'delete-results-id': delete_results_id,
             'origin-gateway': origin_gateway.replica_id, 'target-gateway': target_gateway.replica_id},
            deployment.deployment_ranking.get_first(), deployment)

        time_time = self.now()

        scale_schedule_event = ScaleScheduleEvent(
            ts=time_time,
            fn=deployment.name,
            replica=replica,
            origin_zone=self.cluster,
            delete=False
        )

        return scale_schedule_event

    def get_scale_down_actions(self, pressure_values: pd.DataFrame, ctx: PlatformContext) -> List[ScaleScheduleEvent]:
        """
         Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
         """
        not_under_pressure = self.is_not_under_pressure(pressure_values, ctx)
        logger.info("not under pressure_gateway %d", len(not_under_pressure))
        result = self.teardown_policy(ctx, not_under_pressure)
        return result

    def is_under_pressure(self, pressure_per_gateway: pd.DataFrame, ctx: PlatformContext) -> List[
        Tuple[FunctionReplica, FunctionDeployment, FunctionReplica]]:
        """
       Checks for each gateway and deployment if its current pressure violates the threshold
       :param ctx
       :param result
       :return: gateway and deployment tuples that are under pressure
       """
        under_pressure = []
        if len(pressure_per_gateway) == 0:
            logger.info("No pressure values")
            return []
        # reduce df to look at the mean pressure over all clients per  function and zone
        gateway_node = ctx.node_service.find(self.gateway.node.name)
        zone = gateway_node.labels[zone_label]
        for deployment in ctx.deployment_service.get_deployments():
            for client_zone in ctx.zone_service.get_zones():
                # client zone = x
                # check if pressure from x on a is too high, if yes -> try to schedule instance in x!
                try:
                    mean_pressure = pressure_per_gateway.loc[deployment.fn_name].loc[zone].loc[client_zone][
                        'pressure']
                    if mean_pressure > self.parameters.max_threshold:
                        # at this point we know where the origin for the high pressure comes from
                        target_gateway = ctx.replica_service.find_function_replicas_with_labels(
                            labels={pod_type_label: api_gateway_type_label},
                            node_labels={zone_label: client_zone})[0]
                        under_pressure.append((target_gateway, deployment, self.gateway))
                except KeyError:
                    pass
        return under_pressure

    def is_not_under_pressure(
            self, pressure_values: pd.DataFrame, ctx: PlatformContext
    ) -> List[Tuple[FunctionReplica, FunctionDeployment]]:
        """
        Checks for each gateway if its current pressure violates the threshold
        Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
        :param gateways: gateways to check the pressure
        :param threshold: pressure threshold
        :return: gateways that are under pressure
        """

        not_under_pressure = []
        # reduce df to look at the mean pressure over all clients per  function and zone
        if len(pressure_values) == 0:
            logger.info("No pressure values")
            return []

        pressure_df = pressure_values.groupby(['fn', 'fn_zone']).mean()
        gateway = self.gateway
        gateway_node = ctx.node_service.find(gateway.node.name)
        zone = gateway_node.labels[zone_label]
        for deployment in ctx.deployment_service.get_deployments():
            try:
                mean_pressure = pressure_df.loc[deployment.fn_name].loc[zone]['pressure']
                if mean_pressure < self.parameters.min_threshold:
                    pending_pods = ctx.replica_service.find_function_replicas_with_labels(
                        labels={
                            function_label: deployment.fn_name,
                        },
                        node_labels={
                            zone_label: zone
                        },
                        running=False,
                        state=pod_pending
                    )
                    if len(pending_pods) > 0:
                        logger.info(
                            f"Wanted to scale down FN {deployment.fn_name} in zone {zone}, but had pending pods.")
                    else:
                        not_under_pressure.append((gateway, deployment))
            except KeyError:
                if deployment.labels.get(function_label, None) is not None:
                    logger.info(f'No pressure values found for {zone} - {deployment} - try to shut down')
                    not_under_pressure.append((gateway, deployment))
        return not_under_pressure

    def teardown_policy(
            self,
            ctx: PlatformContext,
            scale_functions: List[Tuple[FunctionReplica, FunctionDeployment]],
    ) -> List[ScaleScheduleEvent]:
        """
        Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
        """
        scale_schedule_events = []
        backup = {}
        for event in scale_functions:
            deployment = event[1]
            fn = deployment.labels[function_label]
            all_replicas = ctx.replica_service.find_function_replicas_with_labels(
                {function_label: fn})
            backup[fn] = len(all_replicas)

        for event in scale_functions:
            gateway = event[0]
            deployment = event[1]
            fn = deployment.labels[function_label]
            gateway_node = ctx.node_service.find(gateway.node.name)
            zone = gateway_node.labels[zone_label]
            replicas = ctx.replica_service.find_function_replicas_with_labels(
                {function_label: fn}, node_labels={zone_label: zone})
            all_replicas = ctx.replica_service.find_function_replicas_with_labels(
                {function_label: fn})
            if len(replicas) == 0:
                logger.info(
                    f"Wanted to remove container with function {fn} but there is no running container anymore in zone {zone}")
                continue
            if len(all_replicas) == 1:
                logger.info(
                    f"Wanted to remove container with function {fn} but there is only one running container anymore in zone {zone}")
                continue
            if backup[fn] <= 1:
                logger.info(f"Tear down policy wanted to scale down function too many times")
                continue
            remove = self.replica_with_lowest_resource_usage(replicas, ctx)
            if remove is None:
                continue

            backup[fn] -= 1

            ts = time.time()

            event = ScaleScheduleEvent(
                ts=ts,
                fn=fn,
                replica=remove,
                origin_zone=zone,
                dest_zone=zone,
                delete=True
            )
            scale_schedule_events.append(event)
        return scale_schedule_events

    def replica_with_lowest_resource_usage(self, replicas: List[FunctionReplica], ctx: PlatformContext) -> Optional[
        FunctionReplica]:
        cpus = []
        lookback = self.parameters.lookback
        for replica in replicas:
            start = datetime.datetime.now() - datetime.timedelta(seconds=lookback)
            end = datetime.datetime.now()
            try:
                cpu = ctx.telemetry_service.get_replica_cpu(replica.replica_id, start.timestamp(), end.timestamp())[
                    'percentage'].mean()
                cpus.append((replica, cpu))
            except TypeError as e:
                logger.error(e)

        cpus.sort(key=lambda x: x[1])
        return None if len(cpus) == 0 else cpus[0][0]
