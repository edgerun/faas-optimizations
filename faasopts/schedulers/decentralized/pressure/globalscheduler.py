import logging
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from faas.context import PlatformContext
from faas.system import Metrics, FunctionReplicaState, FunctionReplica, FunctionNode
from faas.system.scheduling.decentralized import GlobalScheduler
from faas.util.constant import worker_role_label
from kubernetes.utils import parse_quantity
from skippy.core.utils import parse_size_string

logger = logging.getLogger(__name__)


def extract_zone_out_of_name(name: str):
    pattern = r'zone-(a|b|c)'
    match = re.search(pattern, name)
    if match:
        return match.group(0)
    return None

def is_zero_sum_action(create_events: Dict[str, List[str]], x: ScaleScheduleEvent):
    # see if the events have the same destination zone
    to_create = create_events.get(x.dest_zone, None)
    if to_create is not None:

        # check if the function that is supposed to be deleted also in the list of containers to spawn
        if x.fn in to_create:
            return True

    return False

def zone_has_enough_resources(deployment: Deployment, target_zone: str, ctx: Context) -> bool:
    """
    Evaluates if devices in the target zone has a device that can host the given deployment.
    """
    requested_cpu = deployment.parsed_resource_requests.get('cpu', parse_cpu_millis(default_milli_cpu_request))
    requested_mem = deployment.parsed_resource_requests.get('memory', parse_size_string_to_bytes(default_mem_request))
    for node in ctx.node_service.find_nodes_in_zone(target_zone):
        # multiply by 1000 to get milli cpus
        allocatable_cpu = int(node.allocatable['cpu'].replace('m', '')) * 1000
        allocatable_memory = parse_size_string_to_bytes(node.allocatable['memory'])
        pods = ctx.replica_service.get_pod_containers_on_node(node)
        if len(pods) == int(node.allocatable['pods']):
            continue
        for pod in pods:
            pod_resource_requests = pod.parsed_resource_requests
            cpu_request = pod_resource_requests['cpu']
            memory_request = pod_resource_requests['memory']

            allocatable_cpu -= cpu_request
            allocatable_memory -= memory_request

        if allocatable_cpu >= requested_cpu and allocatable_memory > requested_mem:
            return True
    return False
class PressureGlobalScheduler(GlobalScheduler):
    def __init__(self, scheduler_name: str, storage_local_schedulers: Dict[str, str], ctx: PlatformContext,
                 metrics: Metrics, delay, max_scale, pressure_service: PressureService):
        self.scheduler_name = scheduler_name
        self.ctx = ctx
        self.metrics = metrics
        self.delay = delay
        self.max_scale = max_scale
        self.storage_local_schedulers = storage_local_schedulers
        self.pressure_service = pressure_service
        self.zones = ctx.zone_service.get_zones()

    def __str__(self):
        return f"GlobalScheduler: {self.scheduler_name}"

    def remove_zero_sum_actions(self, create_results: List[ScaleScheduleEvent],
                                delete_results: List[ScaleScheduleEvent]):
        create_events = defaultdict(list)
        for result in create_results:
            create_events[result.dest_zone].append(result.fn)

        # get all delete actions that are not reversing the creation event
        filtered_delete = list(
            filter(lambda x: not is_zero_sum_action(create_events, x), delete_results.scale_schedule_events))
        logger.info(f"zero sum actions were identified: {len(filtered_delete) != delete_results.scale_schedule_events}")
        delete_results.scale_schedule_events = filtered_delete
        return delete_results

    def global_scheduling_policy(
            self, deployment: Deployment, target_gateway: PodContainer, pressure_values_by_fn_by_zone: pd.DataFrame,
            ctx: Context
    ) -> str:
    def global_scheduling_policy(self, replica: FunctionReplica) -> str:
        """
        Finds the region to place the given deployment in.
        Determines the region based on pressure and latency.
        Finds the nearest region (relative to the gateway's region) with the highest pressure.
        The highest pressure indicates that we need more resources there.
        param: pressure_values_by_fn_by_zone:
        Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
        """
        pressure_values_by_fn_by_zone = self.ctx.r
        target_gateway_node = ctx.node_service.find(target_gateway.nodeName)
        target_zone = target_gateway_node.labels[zone_label]
        # in case the initial target zone has enough resources, we can schedule it there
        if zone_has_enough_resources(deployment, target_zone, ctx):
            return target_zone
        else:
            # otherwise we have to go through neighboring zones which can host the app
            l = []
            # Make list of pressure values and accompanying latency to afterwards find the zone where the highest pressure
            # is coming from
            for other_zone in ctx.zone_service.get_zones():
                if other_zone == target_zone:
                    continue
                other_gateway = ctx.pod_service.find_pod_containers_with_labels(
                    {pod_type_label: api_gateway_type_label}, node_labels={zone_label: other_zone})[0]
                other_node = other_gateway.nodeName
                node = target_gateway.nodeName
                latency = ctx.network_service.get_latency(other_node, node)
                df = pressure_values_by_fn_by_zone[pressure_values_by_fn_by_zone['fn'] == deployment.fn_name]
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
                if p_c_x_f < self.parameters.max_threshold:
                    # check if new target has enough resources
                    if zone_has_enough_resources(deployment, target_zone, ctx):
                        return target_zone

            # in case no zone fulfills above requirements, look for nearest that can host
            for t in a:
                target_zone = t[2]
                if zone_has_enough_resources(deployment, target_zone, ctx):
                    return target_zone

            # this error happens in case basically all resources are  used
            raise ValueError(f"No one can host: {deployment.fn_name}")
    def find_cluster(self, replica: FunctionReplica):
        origin_cluster = replica.labels['origin_zone']
        nodes_in_cluster_available = self.get_filtered_nodes_in_cluster(replica, origin_cluster)
        if len(nodes_in_cluster_available) == 0:
            nodes_available, pod_per_node = self.get_filtered_nodes(replica)
            logger.info(pod_per_node)

            pod_per_zone = defaultdict(int)
            if len(nodes_available) > 0:
                cpu_dict = defaultdict(list)
                for node, has_enough in nodes_available:
                    zone = extract_zone_out_of_name(node)
                    pod_per_zone[zone] = pod_per_zone[zone] + pod_per_node[node]
                    logger.info(f'{node} - {zone} - {pod_per_node[node]} - {pod_per_zone[zone]}')
                    if not has_enough:
                        continue
                    cpu_dict[zone].append((node, pod_per_node[node]))

                zone_distances: List[Tuple[str, float]] = self.create_network_distances(origin_cluster)
                zone_distances = sorted(zone_distances, key=lambda x: x[1])
                for zone, distance in zone_distances:
                    cpus = cpu_dict[zone]
                    if len(cpus) > 0:
                        logger.info(f'Found scheduler: {zone} - {distance}')
                        found_scheduler = self.storage_local_schedulers[zone]
                        return found_scheduler, zone
                return '', ''

            else:
                logger.error(f'No node found for replica {replica}')
                return '', ''
        else:
            found_scheduler = self.storage_local_schedulers[origin_cluster]
            logger.info("found scheduler: {}".format(found_scheduler))
            return found_scheduler, origin_cluster

    #
    def nodes_available(self, min_cores_required: int, min_memory_required: int):
        ready_nodes = []
        start_ts = time.time()
        pod_per_node = {}
        nodes: List[FunctionNode] = self.ctx.node_service.get_nodes()
        for n in nodes:
            if n.labels.get(worker_role_label) is None:
                continue
            node_name = n.name
            cpu_reserved = 0
            memory_reserved = 0
            count_replicas = 0
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
                count_replicas = count_replicas + 1
            pod_per_node[node_name] = count_replicas
            node = self.ctx.node_service.find(node_name)
            node_cores = node.cpus
            node_memory = node.allocatable['memory']
            if type(node_memory) is str:
                node_memory = parse_size_string(node_memory)
            enough_memory = (memory_reserved + min_memory_required) < node_memory
            enough_cores = cpu_reserved + min_cores_required < node_cores

            has_enough_resources = enough_cores and enough_memory
            if has_enough_resources:
                ready_nodes.append((node_name, True))
            else:
                ready_nodes.append((node_name, False))
        logger.info(ready_nodes)
        end_ts = time.time()
        self.metrics.log('nodes-available', end_ts - start_ts)
        return ready_nodes, pod_per_node

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

    def get_filtered_nodes(self, replica: FunctionReplica):
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
        nodes_available = self.nodes_available(required_cores, required_memory)
        return nodes_available

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

    def create_network_distances(self, origin_cluster: str) -> List[Tuple[str, float]]:
        distances = []
        origin_node = self.ctx.node_service.find_nodes_in_zone(origin_cluster)[0]
        for zone in self.zones:
            nodes = self.ctx.node_service.find_nodes_in_zone(zone)
            node = nodes[0]
            latency = self.ctx.network_service.get_latency(node.name, origin_node.name)
            distances.append((zone, latency))
        return distances
