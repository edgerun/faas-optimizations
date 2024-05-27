import abc
import datetime
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional

import pandas as pd
from faas.context import PlatformContext, FunctionReplicaFactory
from faas.system import Metrics, FunctionReplica, FunctionDeployment
from faas.system.scheduling.decentralized import GlobalScheduler, BaseGlobalSchedulerConfiguration
from faas.util.constant import zone_label, pod_type_label, api_gateway_type_label, function_label, \
    pod_pending

from faasopts.utils.infrastructure.filter import get_filtered_nodes_in_zone
from faasopts.utils.pressure.api import PressureScaleScheduleEvent, PressureAutoscalerParameters
from faasopts.utils.pressure.calculation import identify_above_max_pressure_deployments, PressureResult, \
    remove_zero_sum_actions, prepare_pressure_scale_schedule_events

logger = logging.getLogger(__name__)


class ScaleScheduleEventHandler(abc.ABC):
    def handle(self, scale_event: List[PressureScaleScheduleEvent]):
        raise NotImplementedError()


@dataclass
class PressureGlobalSchedulerConfiguration(BaseGlobalSchedulerConfiguration):
    # key: zone, value: parameters
    parameters: Dict[str, PressureAutoscalerParameters]

    def copy(self):
        copied_parameters = {}
        for zone, params in self.parameters.items():
            copied_parameters[zone] = params.copy()
        return PressureGlobalSchedulerConfiguration(self.scheduler_name, self.scheduler_type, self.delay,
                                                    copied_parameters)


class PressureGlobalScheduler(GlobalScheduler):
    def __init__(self, config: PressureGlobalSchedulerConfiguration, storage_local_schedulers: Dict[str, str],
                 ctx: PlatformContext,
                 metrics: Metrics, now: Callable[[], float],
                 replica_factory: FunctionReplicaFactory,
                 ):
        super().__init__(config)
        self.running = True
        self.ctx = ctx
        self.metrics = metrics
        self.storage_local_schedulers = storage_local_schedulers
        self.zones = ctx.zone_service.get_zones()
        self.parameters = config.parameters
        self.now = now
        self.replica_factory = replica_factory

    def __str__(self):
        return f"GlobalScheduler: {self.scheduler_name}"

    def find_clusters_for_autoscaler_decisions(self, pressure_values: pd.DataFrame) -> List[PressureScaleScheduleEvent]:
        delete_results = self.get_scale_down_actions(pressure_values)

        teardowns = defaultdict(int)
        for result in delete_results:
            teardowns[result.fn] += 1

        # Figure out up scaling actions, previous generated ids for intermediate results are passed in each replica as
        # labels
        create_results = self.get_scale_up_actions(pressure_values, teardowns)
        delete_results = remove_zero_sum_actions(create_results, delete_results)

        logger.info(f"figured out scaling, {len(create_results)} up scale events")
        logger.info("figure our scale down")
        logger.info(f"figured out down scaling, {len(delete_results)} down scale events")
        all_results = []
        all_results.extend(delete_results)
        all_results.extend(create_results)
        return all_results

    def get_scale_up_actions(self, pressure_values: pd.DataFrame, teardowns: Dict[str, int]) -> List[
        PressureScaleScheduleEvent]:
        """
        Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
        """
        under_pressure_new = identify_above_max_pressure_deployments(pressure_values, teardowns, self.ctx,
                                                                     self.parameters)
        scale_schedule_events = self.scheduling_policy(under_pressure_new, pressure_values)

        return scale_schedule_events

    def scheduling_policy(
            self,
            scale_functions: List[PressureResult],
            pressure_per_zone: pd.DataFrame
    ) -> List[PressureScaleScheduleEvent]:
        """
        Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
        """
        events = []
        for event in scale_functions:
            pressure_target_gateway = event.target_gateway
            deployment = event.deployment
            scale_up_rate = 1

            scheduler, new_target_zone = self.global_scheduling_policy(deployment, pressure_target_gateway,
                                                                       pressure_per_zone)
            local_scheduler_name = self.storage_local_schedulers[new_target_zone]
            event = prepare_pressure_scale_schedule_events(deployment, new_target_zone=new_target_zone,
                                                           pressure_target_zone=pressure_target_gateway.node.labels[
                                                               zone_label],
                                                           local_scheduler_name=local_scheduler_name,
                                                           replica_factory=self.replica_factory, now=self.now,
                                                           no_of_replicas=scale_up_rate)
            events.append(event)
        return events

    def get_scale_down_actions(self, pressure_values: pd.DataFrame) -> List[PressureScaleScheduleEvent]:
        """
         Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
         """
        not_under_pressure = self.is_not_under_pressure(pressure_values, self.ctx)
        logger.info("not under pressure_gateway %d", len(not_under_pressure))
        result = self.teardown_policy(self.ctx, not_under_pressure)
        return result

    def is_not_under_pressure(
            self, pressure_values: pd.DataFrame, ctx: PlatformContext
    ) -> List[Tuple[FunctionReplica, FunctionDeployment]]:
        """
        Checks for each gateway if its current pressure violates the threshold
        Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
        :return: gateways that are under pressure
        """

        not_under_pressure = []
        # reduce df to look at the mean pressure over all clients per  function and zone
        if len(pressure_values) == 0:
            logger.info("No pressure values")
            return []

        pressure_df = pressure_values.groupby(['fn', 'fn_zone']).mean()
        for gateway in self.ctx.replica_service.find_function_replicas_with_labels(
                {pod_type_label: api_gateway_type_label}):
            gateway_node = gateway.node
            zone = gateway_node.labels[zone_label]
            for deployment in ctx.deployment_service.get_deployments():
                try:
                    mean_pressure = pressure_df.loc[deployment.name].loc[zone]['pressure']
                    if len(pressure_df.loc[deployment.name]) == 0 or len(
                            pressure_df.loc[deployment.name].loc[zone]) == 0:
                        continue
                    if mean_pressure < self.parameters[zone].function_parameters[deployment.name].min_threshold:
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
                                f"Wanted to scale down FN {deployment.name} in zone {zone}, but had pending pods.")
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
    ) -> List[PressureScaleScheduleEvent]:
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

            event = PressureScaleScheduleEvent(
                ts=ts,
                fn=fn,
                replicas=[remove],
                origin_zone=zone,
                target_zone=zone,
                delete=True
            )
            scale_schedule_events.append(event)
        return scale_schedule_events

    def replica_with_lowest_resource_usage(self, replicas: List[FunctionReplica], ctx: PlatformContext) -> Optional[
        FunctionReplica]:
        cpus = []
        for replica in replicas:
            lookback = self.parameters[replica.labels[zone_label]].function_parameters[replica.function.name].lookback
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

    def global_scheduling_policy(self, deployment: FunctionDeployment, target_gateway: FunctionReplica,
                                 pressure_per_zone: pd.DataFrame) -> Tuple[str, str]:

        replica = self.replica_factory.create_replica({}, deployment.deployment_ranking.get_first(), deployment)

        pressure_values_by_fn_by_zone = pressure_per_zone

        target_gateway_node = target_gateway.node
        target_zone = target_gateway_node.labels[zone_label]

        # in case the initial target zone has enough resources, we can schedule it there
        nodes_in_cluster_available = get_filtered_nodes_in_zone(self.ctx, replica, target_zone)
        if len(nodes_in_cluster_available) > 0:
            found_scheduler = self.storage_local_schedulers[target_zone]
            logger.info("found scheduler: {} for replica {}".format(found_scheduler, replica.replica_id))
            return found_scheduler, target_zone
        else:
            # otherwise we have to go through neighboring zones which can host the app
            l = []
            # Make list of pressure values and accompanying latency to afterward find the zone where the highest pressure
            # is coming from
            for other_zone in self.ctx.zone_service.get_zones():
                if other_zone == target_zone:
                    continue
                other_gateway = self.ctx.replica_service.find_function_replicas_with_labels(
                    {pod_type_label: api_gateway_type_label}, node_labels={zone_label: other_zone})[0]
                other_node = other_gateway.nodeName
                node = target_gateway.node.name
                latency = self.ctx.network_service.get_latency(other_node, node)
                df = pressure_values_by_fn_by_zone[pressure_values_by_fn_by_zone['fn'] == deployment.name]
                df = df[df['client_zone'] == target_zone]
                df = df[df['fn_zone'] == other_zone]
                if len(df) == 0:
                    # we ignore compute units in this step that don't have a pressure, these units are considered
                    # further down in case no zone fulfills requirements
                    continue
                else:
                    p_c_x_f = df['pressure'].iloc[0]
                l.append((latency, p_c_x_f, other_zone))

            # sort by latency
            a = sorted(l, key=lambda k: (k[0]))
            for t in a:
                p_c_x_f = t[1]
                target_zone = t[2]
                # check if pressure is already violated
                if p_c_x_f < self.parameters[target_zone].function_parameters[deployment.name].max_threshold:
                    # check if new target has enough resources
                    if len(get_filtered_nodes_in_zone(self.ctx, replica, target_zone)) > 0:
                        found_scheduler = self.storage_local_schedulers[target_zone]
                        logger.info("found scheduler: {} for replica {}".format(found_scheduler, replica.replica_id))
                        return found_scheduler, target_zone

            # in case no zone fulfills above requirements, look for nearest that can host
            for t in a:
                target_zone = t[2]
                if len(get_filtered_nodes_in_zone(self.ctx, replica, target_zone)):
                    found_scheduler = self.storage_local_schedulers[target_zone]
                    logger.info("found scheduler: {} for replica {}".format(found_scheduler, replica.replica_id))
                    return found_scheduler, target_zone

            # this error happens in case basically all resources are  used
            return '', ''
