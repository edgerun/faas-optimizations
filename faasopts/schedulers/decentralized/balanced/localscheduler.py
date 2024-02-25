import logging
import math
import time
from typing import List

from faas.context import PlatformContext
from faas.system import Metrics, FunctionReplica, FunctionNode
from faas.system.scheduling.decentralized import LocalScheduler
from faas.util.constant import client_role_label, worker_role_label, controller_role_label
from kubernetes.utils import parse_quantity
from skippy.core.utils import parse_size_string

logger = logging.getLogger(__name__)


def busy(delay):
    usage = 0.8
    sleep = 1 - usage
    start_ts = time.time()
    while True:
        startTime = time.time()
        while time.time() - startTime < usage:
            math.factorial(100000)
        time.sleep(sleep)
        if time.time() - start_ts > delay:
            return


def has_valid_role_label(replica: FunctionReplica, node: FunctionNode) -> bool:
    if replica.labels.get(client_role_label) == 'true':
        return node.labels.get(client_role_label) == 'true'

    if replica.labels.get(worker_role_label) == 'true':
        return node.labels.get(worker_role_label) == 'true'

    if replica.labels.get(controller_role_label) == 'true':
        return node.labels.get(controller_role_label) == 'true'

    return True


class LocalBalancedScheduler(LocalScheduler):
    def __init__(self, scheduler_name: str, cluster: str, ctx: PlatformContext, metrics: Metrics,
                 global_scheduler_name="", delay=0):
        self.scheduler_name = scheduler_name
        self.cluster = cluster
        self.ctx = ctx
        self.metrics = metrics
        self.delay = delay
        self.global_scheduler_name = global_scheduler_name

    def __str__(self):
        return f"Scheduler: {self.scheduler_name}"

    def schedule(self, replica: FunctionReplica) -> str:
        nodes_available = self.get_filtered_nodes_in_cluster(replica, self.cluster)
        if len(nodes_available) > 0:
            nodes_available, pod_per_node = self.get_filtered_nodes(replica)
            pods_per_node = []
            for node, has_enough in nodes_available:
                if not has_enough:
                    continue
                pods_per_node.append((node, pod_per_node[node]))
            pods_per_node = sorted(pods_per_node, key=lambda x: x[1])
            return pods_per_node[0][0]
        return None

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

    def nodes_available(self, min_cores_required: int, min_memory_required: int):
        ready_nodes = []
        start_ts = time.time()
        pod_per_node = {}
        nodes: List[FunctionNode] = self.ctx.node_service.find_nodes_in_zone(self.cluster)
        for n in nodes:
            if n.labels.get(worker_role_label) is None:
                continue
            node_name = n.name
            cpu_reserved = 0
            memory_reserved = 0
            count_replicas = 0
            for replica in self.ctx.replica_service.get_function_replicas_on_node(node_name, None):
                # if replica.state != FunctionReplicaState.RUNNING:
                #     continue

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
            logger.info(f'Node: {node_name}, CPU reserved: {cpu_reserved}, min_cores_required: {min_cores_required}, node_cores: {node_cores}')
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
                # if replica.state != FunctionReplicaState.RUNNING:
                #     continue
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
        cpu_request /= 1000
        memory_request = resources_requests.get('memory')

        nodes_available = self.nodes_available_in_cluster(cpu_request, memory_request, cluster)
        return nodes_available
