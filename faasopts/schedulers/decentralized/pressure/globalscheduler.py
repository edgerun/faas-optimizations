import abc
import datetime
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional

import pandas as pd
from faas.context import PlatformContext, FunctionReplicaFactory
from faas.system import Metrics, FunctionReplicaState, FunctionReplica, FunctionNode, FunctionDeployment
from faas.system.scheduling.decentralized import GlobalScheduler, BaseGlobalSchedulerConfiguration
from faas.util.constant import worker_role_label, zone_label, pod_type_label, api_gateway_type_label, function_label, \
    pod_pending
from kubernetes.utils import parse_quantity
from skippy.core.utils import parse_size_string

from faasopts.utils.pressure.api import ScaleScheduleEvent, PressureAutoscalerParameters
from faasopts.utils.pressure.service import PressureService

logger = logging.getLogger(__name__)



def is_zero_sum_action(create_events: Dict[str, List[str]], x: ScaleScheduleEvent):
    # see if the events have the same destination zone
    to_create = create_events.get(x.target_zone, None)
    if to_create is not None:

        # check if the function that is supposed to be deleted also in the list of containers to spawn
        if x.fn in to_create:
            return True

    return False


class ScaleScheduleEventHandler(abc.ABC):
    def handle(self, scale_event: List[ScaleScheduleEvent]):
        raise NotImplementedError()


@dataclass
class PressureGlobalSchedulerConfiguration(BaseGlobalSchedulerConfiguration):
    # key: cluster, value: parameters
    parameters: Dict[str, PressureAutoscalerParameters]
    scale_schedule_event_handler: ScaleScheduleEventHandler

class PressureGlobalScheduler(GlobalScheduler):
    def __init__(self, config: PressureGlobalSchedulerConfiguration, storage_local_schedulers: Dict[str, str],
                 ctx: PlatformContext,
                 metrics: Metrics, pressure_service: PressureService, now: Callable[[], float],
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
        self.scale_schedule_event_handler = config.scale_schedule_event_handler
        self.pressure_service = pressure_service

    def __str__(self):
        return f"GlobalScheduler: {self.scheduler_name}"

    def remove_zero_sum_actions(self, create_results: List[ScaleScheduleEvent],
                                delete_results: List[ScaleScheduleEvent]):
        """
        Removes zero sum actions from create_results and delete_results
        :param create_results: tuples of replicas and target zones
        :param delete_results: delete actions to consider
        :return:
        """
        create_events = defaultdict(list)
        for result in create_results:
            create_events[result.target_zone].append(result.fn)

        # get all delete actions that are not reversing the creation event
        filtered_delete = list(
            filter(lambda x: not is_zero_sum_action(create_events, x), delete_results))
        logger.info(f"zero sum actions were identified: {len(filtered_delete) != delete_results}")
        return filtered_delete

    def run(self):
        while self.running:
            # wait for all autoscalers to finish
            pressure_values: pd.DataFrame = self.pressure_service.wait_for_all_pressures()
            scale_schedule_events = self.find_clusters_for_autoscaler_decisions(pressure_values)
            self.scale_schedule_event_handler.handle(scale_schedule_events)

    def find_clusters_for_autoscaler_decisions(self, pressure_values: pd.DataFrame) -> List[ScaleScheduleEvent]:

        # Figure out down scaling actions and save them for global scheduler
        delete_results = self.get_scale_down_actions(pressure_values)

        # Figure out up scaling actions, previous generated ids for intermediate results are passed in each replica as
        # labels
        create_results = self.get_scale_up_actions(pressure_values, len(delete_results))
        delete_results = self.remove_zero_sum_actions(create_results, delete_results)

        logger.info(f"figured out scaling, {len(create_results)} up scale events")
        logger.info("figure our scale down")
        logger.info(f"figured out down scaling, {len(delete_results)} down scale events")
        all_results = []
        all_results.extend(delete_results)
        all_results.extend(create_results)
        return all_results

    def get_scale_up_actions(self, pressure_values: pd.DataFrame, teardowns: int) -> List[
        ScaleScheduleEvent]:
        """
        Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
        """
        under_pressures = self.is_under_pressure(pressure_values)
        logger.info(" under pressure_gateway %d", len(under_pressures))
        under_pressure_new = []
        new_pods = {}
        for under_pressure in under_pressures:
            deployment = under_pressure[1]
            max_replica = self.parameters[deployment.name].max_replicas
            running_pods = len(self.ctx.replica_service.get_function_replicas_of_deployment(deployment.name))
            pending_pods = len(
                self.ctx.replica_service.get_function_replicas_of_deployment(deployment.name, running=False,
                                                                             state=pod_pending))
            all_pods = running_pods + pending_pods
            no_new_pods = new_pods.get(deployment.name, 0)
            if ((all_pods + no_new_pods) - teardowns) < max_replica:
                under_pressure_new.append(under_pressure)
                if new_pods.get(deployment.name, None) is None:
                    new_pods[deployment.name] = 1
                else:
                    new_pods[deployment.name] += 1

        scale_schedule_events = self.scheduling_policy(under_pressure_new, pressure_values)

        return scale_schedule_events

    def scheduling_policy(
            self,
            scale_functions: List[Tuple[FunctionReplica, FunctionDeployment, FunctionReplica]],
            pressure_per_zone: pd.DataFrame
    ) -> List[ScaleScheduleEvent]:
        """
        Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
        """
        events = []
        for event in scale_functions:
            pressure_target_gateway = event[0]
            deployment = event[1]
            scheduler, new_target_zone = self.global_scheduling_policy(deployment, pressure_target_gateway,
                                                                       pressure_per_zone)

            event = self.prepare_pod_request(deployment, new_target_zone=new_target_zone,
                                             pressure_target_zone=pressure_target_gateway.node.labels[zone_label])
            events.append(event)
        return events

    def prepare_pod_request(self, deployment: FunctionDeployment, new_target_zone: str,
                            pressure_target_zone: str) -> ScaleScheduleEvent:
        local_scheduler_name = self.storage_local_schedulers[new_target_zone]
        replica = self.replica_factory.create_replica(
            {worker_role_label: 'true', 'origin_zone': pressure_target_zone,
             zone_label: new_target_zone, 'schedulerName': local_scheduler_name},
            deployment.deployment_ranking.get_first(), deployment)

        time_time = self.now()

        scale_schedule_event = ScaleScheduleEvent(
            ts=time_time,
            fn=deployment.name,
            replica=replica,
            origin_zone=pressure_target_zone,
            target_zone=new_target_zone,
            delete=False
        )

        return scale_schedule_event

    def get_scale_down_actions(self, pressure_values: pd.DataFrame) -> List[ScaleScheduleEvent]:
        """
         Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
         """
        not_under_pressure = self.is_not_under_pressure(pressure_values, self.ctx)
        logger.info("not under pressure_gateway %d", len(not_under_pressure))
        result = self.teardown_policy(self.ctx, not_under_pressure)
        return result

    def is_under_pressure(self, pressure_per_gateway: pd.DataFrame) -> List[
        Tuple[FunctionReplica, FunctionDeployment, FunctionReplica]]:
        """
       Checks for each gateway and deployment if its current pressure violates the threshold
       :return: gateway and deployment tuples that are under pressure
       """
        under_pressure = []
        if len(pressure_per_gateway) == 0:
            logger.info("No pressure values")
            return []
        # reduce df to look at the mean pressure over all clients per  function and zone
        for gateway in self.ctx.replica_service.find_function_replicas_with_labels(
                {pod_type_label: api_gateway_type_label}):
            gateway_node = gateway.node
            zone = gateway_node.labels[zone_label]
            for deployment in self.ctx.deployment_service.get_deployments():
                for client_zone in self.ctx.zone_service.get_zones():
                    # client zone = x
                    # check if pressure from x on a is too high, if yes -> try to schedule instance in x!
                    try:
                        mean_pressure = pressure_per_gateway.loc[deployment.name]
                        if len(mean_pressure) == 0 or len(mean_pressure.loc[client_zone]) == 0:
                            continue
                        mean_pressure = mean_pressure.loc[zone].loc[client_zone][
                            'pressure']
                        if mean_pressure > self.parameters[deployment.name].max_threshold:
                            # at this point we know where the origin for the high pressure comes from
                            target_gateway = self.ctx.replica_service.find_function_replicas_with_labels(
                                labels={pod_type_label: api_gateway_type_label},
                                node_labels={zone_label: client_zone})[0]
                            under_pressure.append((target_gateway, deployment, gateway))
                    except KeyError:
                        pass
        return under_pressure

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
                    if mean_pressure < self.parameters[deployment.name].min_threshold:
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
                delete=True
            )
            scale_schedule_events.append(event)
        return scale_schedule_events

    def replica_with_lowest_resource_usage(self, replicas: List[FunctionReplica], ctx: PlatformContext) -> Optional[
        FunctionReplica]:
        cpus = []
        lookback = self.parameters[replicas[0].function.name].lookback
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

    def nodes_available_in_cluster(self, min_cores_required: int, min_memory_required: int, cluster: str) -> List[str]:
        ready_nodes = []
        start_ts = time.time()
        nodes: List[FunctionNode] = self.ctx.node_service.get_nodes()
        for n in nodes:
            node_cluster_label = n.cluster
            if node_cluster_label != cluster or n.labels.get(worker_role_label) is None:
                continue
            node_name = n.name
            cpu_reserved = 0
            memory_reserved = 0
            for replica in self.ctx.replica_service.get_function_replicas_on_node(node_name, None):
                if replica.state == FunctionReplicaState.DELETE or replica.state == FunctionReplicaState.SHUTDOWN:
                    continue
                cpu = replica.container.get_resource_requirements()['cpu']
                if type(cpu) is str:
                    cpu = parse_quantity(cpu)
                else:
                    cpu /= 1000
                cpu_reserved += cpu
                memory = replica.container.get_resource_requirements()['memory']
                if type(memory) is str:
                    memory = parse_size_string(memory)
                memory_reserved += memory

            node = self.ctx.node_service.find(node_name)
            node_cores = node.cpus
            node_memory = node.allocatable['memory']
            if type(node_memory) is str:
                node_memory = parse_size_string(node_memory)

            enough_memory = (memory_reserved + min_memory_required) < node_memory
            enough_cores = cpu_reserved + min_cores_required < node_cores
            has_enough_resources = enough_cores and enough_memory
            if has_enough_resources:
                ready_nodes.append(node_name)
        logger.info(ready_nodes)
        end_ts = time.time()
        self.metrics.log('nodes-available', end_ts - start_ts)
        return ready_nodes

    def get_filtered_nodes_in_cluster(self, replica: FunctionReplica, cluster: str):
        resources_requests = replica.container.get_resource_requirements()
        cpu_request = resources_requests.get('cpu')
        if cpu_request and type(cpu_request) is cpu_request:
            required_cores = parse_quantity(cpu_request)
        elif not cpu_request:
            required_cores = 0
        else:
            required_cores = cpu_request
        required_cores /= 1000
        memory_request = resources_requests.get('memory')
        if memory_request and type(memory_request) is str:
            required_memory = parse_size_string(memory_request)
        elif not memory_request:
            required_memory = 0
        else:
            required_memory = memory_request
        nodes_available = self.nodes_available_in_cluster(required_cores, required_memory, cluster)
        return nodes_available

    def get_pressure_values(self, pressure_id: str) -> pd.DataFrame:
        return self.pressure_service.get_pressure_values(pressure_id)

    def get_delete_results(self, delete_results_id: str) -> List[ScaleScheduleEvent]:
        raise self.pressure_service.get_delete_results(delete_results_id)

    def global_scheduling_policy(self, deployment: FunctionDeployment, target_gateway: FunctionReplica,
                                 pressure_per_zone: pd.DataFrame) -> Tuple[str, str]:

        replica = self.replica_factory.create_replica({}, deployment.deployment_ranking.get_first(), deployment)

        pressure_values_by_fn_by_zone = pressure_per_zone

        target_gateway_node = target_gateway.node
        target_zone = target_gateway_node.labels[zone_label]

        # in case the initial target zone has enough resources, we can schedule it there
        nodes_in_cluster_available = self.get_filtered_nodes_in_cluster(replica, target_zone)
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
                if p_c_x_f < self.parameters[deployment.name].max_threshold:
                    # check if new target has enough resources
                    if len(self.get_filtered_nodes_in_cluster(replica, target_zone)) > 0:
                        found_scheduler = self.storage_local_schedulers[target_zone]
                        logger.info("found scheduler: {} for replica {}".format(found_scheduler, replica.replica_id))
                        return found_scheduler, target_zone

            # in case no zone fulfills above requirements, look for nearest that can host
            for t in a:
                target_zone = t[2]
                if len(self.get_filtered_nodes_in_cluster(replica, target_zone)):
                    found_scheduler = self.storage_local_schedulers[target_zone]
                    logger.info("found scheduler: {} for replica {}".format(found_scheduler, replica.replica_id))
                    return found_scheduler, target_zone

            # this error happens in case basically all resources are  used
            return '', ''

    def handle_scale_schedule_events(self, scale_schedule_events: List[ScaleScheduleEvent]):
        self.scale_schedule_event_handler.handle(scale_schedule_events)
