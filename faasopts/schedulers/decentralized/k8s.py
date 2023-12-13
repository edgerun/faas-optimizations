import json
import logging
import math
import re
import signal
import sys
import time
from statistics import mean
from typing import Dict, List, Any, Union

from faas.context import PlatformContext
from faas.system import Metrics, FunctionReplicaState, FaasSystem
from faas.util.constant import zone_label
from skippy.core.utils import parse_size_string


logger = logging.getLogger(__name__)


def extract_zone_out_of_name(name: str):
    pattern = r'zone-(a|b|c)'
    match = re.search(pattern, name)
    if match:
        return match.group(0)
    return None


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


class K8sGlobalScheduler:
    def __init__(self, scheduler_name: str, storage_local_schedulers: Dict[str, str], ctx: PlatformContext,
                 faas: FaasSystem, global_scheduler: GlobalScheduler,
                 metrics: Metrics, delay, max_scale):
        self.scheduler_name = scheduler_name
        self.ctx = ctx
        self.metrics = metrics
        self.delay = delay
        self.max_scale = max_scale
        self.storage_local_schedulers = storage_local_schedulers
        self.zones = [zone for zone in storage_local_schedulers.keys()]
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def __str__(self):
        return f"Scheduler: {self.scheduler_name}"

    def signal_handler(self, signal, frame):
        print('Signal received!')
        self.create_poison_pod_for_scheduler(self.scheduler_name)

    def create_poison_pod_for_scheduler(self, scheduler_name: str):
        pod_metadata = client.V1ObjectMeta()
        pod_metadata.name = f'poison-pod-{scheduler_name}'
        pod_metadata.labels = {'app': 'poison-pod'}

        container = client.V1Container(name='empty', image='alpine')
        pod_spec = client.V1PodSpec(containers=[container])
        pod_spec.scheduler_name = scheduler_name
        pod_body = client.V1Pod(metadata=pod_metadata, spec=pod_spec, kind='Pod', api_version='v1')

        self.v1.create_namespaced_pod(namespace='default', body=pod_body)


    def find_most_suitable_scheduler_for_pod(self, event):
        # TODO: das ist ja immer noch random choice?
        pod: V1Pod = event['object']
        origin_cluster = pod.metadata.labels['origin_zone']
        nodes_in_cluster_available = self.get_filtered_nodes_in_cluster(pod, origin_cluster)
        if len(nodes_in_cluster_available) == 0:
            nodes_available, pod_per_node = self.get_filtered_nodes(pod)
            logger.info(pod_per_node)

            pod_per_zone = {'zone-a': 0, 'zone-b': 0, 'zone-c': 0}
            if len(nodes_available) > 0:
                cpu_dict = {'zone-a': [], 'zone-b': [], 'zone-c': []}
                for node, has_enough in nodes_available:
                    zone = extract_zone_out_of_name(node)
                    pod_per_zone[zone] = pod_per_zone[zone] + pod_per_node[node]
                    logger.info(f'{node} - {zone} - {pod_per_node[node]} - {pod_per_zone[zone]}')
                    if not has_enough:
                        continue
                    node_cpu = self.ctx.telemetry_service.get_node_cpu(node)

                    if len(node_cpu) == 0:
                        cpu_dict.get(zone).append(0)
                    else:
                        cpu = node_cpu['value'].mean()
                        cpu_dict.get(zone).append(cpu)
                current_min = sys.float_info.max
                current_min_zone = ''
                print(cpu_dict)
                logger.info(pod_per_zone)
                logger.info(self.max_scale)
                for (zone, cpus) in cpu_dict.items():
                    if pod_per_zone[zone] >= self.max_scale - 1:
                        continue
                    if not cpus:
                        cpus.append(sys.float_info.max)
                        continue
                    if mean(cpus) <= current_min:
                        current_min_zone = zone
                        current_min = mean(cpus)
                logger.info(f'Chose {current_min_zone}')
                if current_min_zone:
                    found_scheduler = self.storage_local_schedulers[current_min_zone]
                    print("found scheduler: {}".format(found_scheduler))
                    return found_scheduler, current_min_zone
                else:
                    self.v1.delete_namespaced_pod(pod.metadata.name, 'default')
                    return '', ''
            else:
                self.v1.delete_namespaced_pod(pod.metadata.name, 'default')
                return '', ''
        else:
            found_scheduler = self.storage_local_schedulers[origin_cluster]
            logger.info("found scheduler: {}".format(found_scheduler))
            return found_scheduler, origin_cluster

    #
    def nodes_available(self, min_cores_required: int, min_memory_required: int) -> tuple[
        list[Any], dict[Any, Union[int, Any]]]:
        ready_nodes = []
        start_ts = time.time()
        pod_per_node = {}
        for n in self.v1.list_node(label_selector='node-role.kubernetes.io/worker=true').items:
            for status in n.status.conditions:
                if status.status == "True" and status.type == "Ready":
                    node_name = n.metadata.name
                    cpu_reserved = 0
                    memory_reserved = 0
                    count_replicas = 0
                    for replica in self.ctx.replica_service.get_function_replicas_on_node(node_name):
                        if replica.state != FunctionReplicaState.RUNNING:
                            continue
                        cpu = parse_quantity(replica.container.get_resource_requirements()['cpu'])
                        cpu_reserved += cpu
                        memory = parse_size_string(replica.container.get_resource_requirements()['memory'])
                        memory_reserved += memory
                        count_replicas = count_replicas + 1
                    pod_per_node[node_name] = count_replicas
                    node = self.ctx.node_service.find(node_name)
                    node_cores = node.cpus
                    node_memory = parse_size_string(node.allocatable['memory'])
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
        for n in self.v1.list_node(label_selector='node-role.kubernetes.io/worker=true').items:
            for status in n.status.conditions:
                if status.status == "True" and status.type == "Ready":
                    node_cluster_label = n.metadata.labels.get(zone_label)
                    if node_cluster_label != cluster:
                        continue
                    node_name = n.metadata.name
                    cpu_reserved = 0
                    memory_reserved = 0
                    for replica in self.ctx.replica_service.get_function_replicas_on_node(node_name):
                        if replica.state != FunctionReplicaState.RUNNING:
                            continue
                        cpu = parse_quantity(replica.container.get_resource_requirements()['cpu'])
                        cpu_reserved += cpu
                        memory = parse_size_string(replica.container.get_resource_requirements()['memory'])
                        memory_reserved += memory

                    node = self.ctx.node_service.find(node_name)
                    node_cores = node.cpus
                    node_memory = parse_size_string(node.allocatable['memory'])
                    enough_memory = (memory_reserved + min_memory_required) < node_memory
                    enough_cores = cpu_reserved + min_cores_required < node_cores
                    has_enough_resources = enough_cores and enough_memory
                    if has_enough_resources:
                        ready_nodes.append(node_name)
        logger.info(ready_nodes)
        end_ts = time.time()
        self.metrics.log('nodes-available', end_ts - start_ts)
        return ready_nodes


    def get_filtered_nodes(self, pod):
        resources_requests = pod.spec.containers[0].resources.requests
        cpu_request = resources_requests.get('cpu')
        if cpu_request:
            required_cores = parse_quantity(cpu_request)
        else:
            required_cores = 0
        memory_request = resources_requests.get('memory')
        if memory_request:
            required_memory = parse_size_string(memory_request)
        else:
            required_memory = 0
        nodes_available = self.nodes_available(required_cores, required_memory)
        return nodes_available

    def get_filtered_nodes_in_cluster(self, pod, cluster: str):
        resources_requests = pod.spec.containers[0].resources.requests
        cpu_request = resources_requests.get('cpu')
        if cpu_request:
            required_cores = parse_quantity(cpu_request)
        else:
            required_cores = 0
        memory_request = resources_requests.get('memory')
        if memory_request:
            required_memory = parse_size_string(memory_request)
        else:
            required_memory = 0
        nodes_available = self.nodes_available_in_cluster(required_cores, required_memory, cluster)
        return nodes_available

    def reschedule_pod(self, pod_name: str, scheduler_name: str, zone: str) -> V1Pod:
        # because schedulerName from pods cannot be changed, so we delete the pod to reschedule
        pod = self.v1.delete_namespaced_pod(name=pod_name, namespace='default')
        last_applied_configuration_json = pod.metadata.annotations['kubectl.kubernetes.io/last-applied-configuration']
        pod_data = json.loads(last_applied_configuration_json)
        pod_data['metadata']['annotations'][
            'kubectl.kubernetes.io/last-applied-configuration'] = last_applied_configuration_json
        pod_data['metadata']['labels']['ether.edgerun.io/zone'] = zone
        pod_spec = pod.spec
        pod_spec.node_selector = {'ether.edgerun.io/zone': zone}
        pod_spec.scheduler_name = scheduler_name
        pod_copy = client.V1Pod(api_version=pod_data['api_version'], kind=pod_data['kind'],
                                metadata=pod_data['metadata'], spec=pod_spec)
        return self.v1.create_namespaced_pod(namespace='default', body=pod_copy)

    def start_schedule(self):
        print("Start scheduling %s" % self.scheduler_name)
        running = True
        while running:
            w = watch.Watch()
            stream = w.stream(self.v1.list_namespaced_pod, namespace="default")
            for event in stream:
                pod: V1Pod = event['object']
                if pod.status.phase == "Pending" and pod.spec.scheduler_name == self.scheduler_name and \
                        pod.spec.node_name is None and pod.metadata.deletion_timestamp is None:

                    if pod.metadata.name == f'poison-pod-{self.scheduler_name}':
                        w.stop()
                        running = False
                        self.v1.delete_namespaced_pod(name=f'poison-pod-{self.scheduler_name}', namespace='default')
                        break
                    if self.delay > 0:
                        busy(self.delay)
                    start_ts = time.time()
                    print("finding best local scheduler for pod: %s" % pod.metadata.name)
                    try:
                        scheduler_for_pod, zone = self.find_most_suitable_scheduler_for_pod(event)
                        if scheduler_for_pod:
                            new_pod = self.reschedule_pod(pod.metadata.name, scheduler_for_pod, zone)
                            print("scheduled")
                            end_ts = time.time()
                            self.metrics.log('global-scheduler-delay', end_ts - start_ts,
                                             pod_name=new_pod.metadata.name, start_ts=start_ts, end_ts=end_ts)
                        else:
                            print('Nothing suitable found')
                            self.v1.delete_namespaced_pod(pod.metadata.name, 'default')
                    except client.exceptions.ApiException as e:
                        print(json.loads(e.body)['message'])

        print("End scheduling %s" % self.scheduler_name)


def main():
    daemon = None
    scheduler = None
    try:
        config.load_kube_config()
        zone = 'zone-c'
        daemon, faas_system, metrics = setup_galileo_faas(f'scheduler-{zone}')
        daemon.start()
        ctx = daemon.context
        # delay = 0.5  # in s
        delay = 0.5  # in s
        storage = {'zone-a': 'scheduler-zone-a', 'zone-b': 'scheduler-zone-b', 'zone-c': 'scheduler-zone-c'}
        scheduler = GlobalScheduler('global-scheduler', storage, ctx, metrics, delay)
        scheduler.start_schedule()
    finally:
        if daemon is not None:
            scheduler.running = False
            daemon.stop(5)


if __name__ == '__main__':
    main()
