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


class LocalityGlobalScheduler(GlobalScheduler):
    def __init__(self, scheduler_name: str, storage_local_schedulers: Dict[str, str], ctx: PlatformContext,
                 metrics: Metrics, delay, max_scale):
        self.scheduler_name = scheduler_name
        self.ctx = ctx
        self.metrics = metrics
        self.delay = delay
        self.max_scale = max_scale
        self.storage_local_schedulers = storage_local_schedulers
        self.zones = ctx.zone_service.get_zones()

    def __str__(self):
        return f"GlobalScheduler: {self.scheduler_name}"

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
