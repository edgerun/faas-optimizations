import json
import logging
import math
import signal
import time
from typing import Optional

from faas.context import PlatformContext
from faas.system import Metrics, FunctionReplicaState, FunctionReplica
from faas.system.scheduling.decentralized import LocalScheduler
from faas.util.constant import function_label
from galileofaas.system.core import KubernetesFunctionReplica
from kubernetes import client, watch
from kubernetes.client import V1Pod

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


class K8sLocalScheduler:
    def __init__(self, scheduler_name: str, zone: str, ctx: PlatformContext, metrics: Metrics, delay,
                 local_scheduler: LocalScheduler, global_scheduler_name=""):
        self.scheduler_name = scheduler_name
        self.v1 = client.CoreV1Api()
        self.zone = zone
        self.ctx = ctx
        self.metrics = metrics
        self.delay = delay
        self.global_scheduler_name = global_scheduler_name
        self.local_scheduler = local_scheduler
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

    def scheduler_pod(self, pod_name: str, node_name: str, namespace="default"):
        target = client.V1ObjectReference()
        target.kind = "Node"
        target.apiVersion = "v1"
        target.name = node_name

        meta = client.V1ObjectMeta()
        meta.name = pod_name

        body = client.V1Binding(target=target, metadata=meta)

        # looks like to be a lib error: https://github.com/kubernetes-client/python/issues/547
        self.v1.create_namespaced_pod_binding(pod_name, namespace, body, _preload_content=False)
        print("scheduled")

    def start_schedule(self):
        print("Start scheduling %s" % self.scheduler_name)
        running = True
        while running:
            w = watch.Watch()
            stream = w.stream(self.v1.list_namespaced_pod, namespace="default",timeout_seconds=0)
            for event in stream:
                pod = event['object']
                if pod.status.phase == "Pending" and pod.spec.scheduler_name == self.scheduler_name and \
                        pod.spec.node_name is None and pod.metadata.deletion_timestamp is None:

                    if pod.metadata.name == f'poison-pod-{self.scheduler_name}':
                        w.stop()
                        running = False
                        self.v1.delete_namespaced_pod(name=f'poison-pod-{self.scheduler_name}', namespace='default')
                        break

                    print("scheduling pod: %s" % pod.metadata.name)
                    if self.delay > 0:
                        busy(self.delay)
                    start_ts = time.time()
                    try:
                        pod: V1Pod = event['object']

                        deployment = self.ctx.deployment_service.get_by_name(pod.metadata.labels[function_label])
                        container = deployment.get_containers()[0]
                        replica = FunctionReplica(
                            pod.metadata.name,
                            pod.metadata.labels,
                            deployment,
                            container,
                            None,
                            FunctionReplicaState.PENDING
                        )
                        node: Optional[str] = self.local_scheduler.schedule(replica)
                        if node:
                            logger.info(f'Chose {node}')
                            replica.node = self.ctx.node_service.find(node)
                            self.add_replica(replica)
                            # self.recreate_pod_as_replica(pod_name=pod.metadata.name, node_name=node)
                            self.scheduler_pod(pod.metadata.name, node, "default")

                            end_ts = time.time()
                            # TODO add more info, if necessary
                            self.metrics.log('local-scheduler-delay', end_ts - start_ts, pod_name=pod.metadata.name,
                                             start_ts=start_ts, end_ts=end_ts)
                        else:
                            print("Could not schedule pod: %s" % pod.metadata.name)
                            self.v1.delete_namespaced_pod(name=pod.metadata.name, namespace='default')
                    except client.exceptions.ApiException as e:
                        print(json.loads(e.body)['message'])

        print("End scheduling %s" % self.scheduler_name)

    def add_replica(self, replica: FunctionReplica):
        replica.state = FunctionReplicaState.RUNNING
        k8s_replica = KubernetesFunctionReplica(
            replica,
            ip=None,
            port=8080,
            url=None,
            namespace='default',
            host_ip=None,
            qos_class=None,
            start_time=None,
            pod_name=replica.replica_id,
            container_id=None
        )
        self.ctx.replica_service.add_function_replica(k8s_replica)
