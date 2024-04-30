import logging
import math
import time
from typing import List

from faas.context import PlatformContext
from faas.system import Metrics, FunctionReplicaState, FunctionReplica, FunctionNode
from faas.system.scheduling.decentralized import LocalScheduler, BaseLocalSchedulerConfiguration
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


class LocalCpuScheduler(LocalScheduler):
    def __init__(self, config: BaseLocalSchedulerConfiguration, ctx: PlatformContext, metrics: Metrics, delay=0):
        super().__init__(config)
        self.ctx = ctx
        self.metrics = metrics
        self.delay = delay

    def __str__(self):
        return f"Scheduler: {self.scheduler_name}"

    def schedule(self, replica: FunctionReplica) -> str:
        nodes_available = self.get_filtered_nodes_in_cluster(replica, self.zone)
        nodes_by_name = self.ctx.node_service.get_nodes_by_name()
        if len(nodes_available) > 0:
            cpus = []
            for node in nodes_available:
                function_node = nodes_by_name[node]
                if not has_valid_role_label(replica, function_node):
                    continue

                node_cpu = self.ctx.telemetry_service.get_node_cpu(node)
                if node_cpu is None or len(node_cpu) == 0:
                    cpus.append((node, 0))
                else:
                    cpu = node_cpu['value'].mean()
                    cpus.append((node, cpu))
            cpus.sort(key=lambda x: x[1])
            logger.info(cpus)
            node = cpus[0][0]
            return node
        return None

    def nodes_available_in_cluster(self, min_cores_required: int, min_memory_required: int, cluster: str) -> List[str]:
        ready_nodes = []
        start_ts = time.time()
        nodes: List[FunctionNode] = self.ctx.node_service.get_nodes()
        for n in nodes:
            node_cluster_label = n.cluster
            if node_cluster_label != cluster:
                continue
            node_name = n.name
            cpu_reserved = 0
            memory_reserved = 0
            for replica in self.ctx.replica_service.get_function_replicas_on_node(node_name):
                if replica.state != FunctionReplicaState.RUNNING:
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
        cpu_request /= 1000
        memory_request = resources_requests.get('memory')

        nodes_available = self.nodes_available_in_cluster(cpu_request, memory_request, cluster)
        return nodes_available
