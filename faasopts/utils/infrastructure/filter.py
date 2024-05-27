import logging
from typing import List

from faas.context import PlatformContext
from faas.system import FunctionReplicaState, FunctionReplica, FunctionNode
from faas.util.constant import worker_role_label
from kubernetes.utils import parse_quantity
from skippy.core.utils import parse_size_string

logger = logging.getLogger(__name__)


def get_filtered_nodes_in_zone(ctx: PlatformContext, replica: FunctionReplica, zone: str):
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
    nodes = nodes_available_in_zone(ctx, required_cores, required_memory, zone)
    return nodes


def nodes_available_in_zone(ctx: PlatformContext, min_cores_required: int, min_memory_required: int, zone: str) -> \
        List[str]:
    ready_nodes = []
    nodes: List[FunctionNode] = ctx.node_service.get_nodes()
    for n in nodes:
        node_cluster_label = n.cluster
        if node_cluster_label != zone or n.labels.get(worker_role_label) is None:
            continue
        node_name = n.name
        cpu_reserved = 0
        memory_reserved = 0
        for replica in ctx.replica_service.get_function_replicas_on_node(node_name, None):
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

        node = ctx.node_service.find(node_name)
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
    return ready_nodes


def nodes_available(ctx: PlatformContext, min_cores_required: int, min_memory_required: int):
    ready_nodes = []
    pod_per_node = {}
    nodes: List[FunctionNode] = ctx.node_service.get_nodes()
    for n in nodes:
        if n.labels.get(worker_role_label) is None:
            continue
        node_name = n.name
        cpu_reserved = 0
        memory_reserved = 0
        count_replicas = 0
        for replica in ctx.replica_service.get_function_replicas_on_node(node_name, None):
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
        node = ctx.node_service.find(node_name)
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
    return ready_nodes, pod_per_node


def get_filtered_nodes(ctx: PlatformContext, replica: FunctionReplica):
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
    nodes = nodes_available(ctx, required_cores, required_memory)
    return nodes
