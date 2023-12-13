import datetime
import logging
import math
from dataclasses import dataclass
from typing import Dict, Callable, Union, List

import numpy as np
from faas.context import PlatformContext, FunctionReplicaService, FunctionDeploymentService, TraceService, \
    ResponseRepresentation, FunctionReplicaFactory
from faas.system import FaasSystem, Metrics, FunctionReplicaState, Clock, FunctionReplica
from faas.util.constant import zone_label, worker_role_label

from faasopts.autoscalers.api import BaseAutoscaler

logger = logging.getLogger(__name__)


@dataclass
class HorizontalLatencyPodAutoscalerParameters:
    # the past (in seconds) that should be considered when looking at monitoring data
    lookback: int

    # either latency (in ms) or rtt (in s)
    target_time_measure: str

    #  the tolerance within the target metric can be without triggering in our out scaling
    threshold_tolerance: float = 0.0
    # ms when latency, s when rtt
    target_duration: float = 100
    # which percentile should be used to calculate ratio
    percentile_duration: float = 90


class DecentralizedHorizontalLatencyPodAutoscaler(BaseAutoscaler):
    """
    This Optimizer implementation is based on the official default Kubernetes HPA and uses latency to determine
    the number of replicas.

    Reference: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
    """

    def __init__(self, parameters: Dict[str, HorizontalLatencyPodAutoscalerParameters], ctx: PlatformContext,
                 faas: FaasSystem,
                 metrics: Metrics, now: Callable[[], float], cluster: str = None,
                 replica_factory: FunctionReplicaFactory = None, use_yield: bool = False):
        """
        Initializes the HPA latency-based implementation
        :param parameters: HPA parameters per deployment that dictate various configuration values
        :param ctx: the HPA will get any information (i.e., monitoring data) from the context
        :param faas: the system will be invoked when scaling in our out
        :param metrics: is used to log any scaling decisions for later analysis
        :param cluster: if set, only looks at replica that reside in the given cluster (ether.edgerun.io/zone label)
        :param replica_factory: if cluster argument is passed, this replica_factory will be used to create replicas with an appropriate set node selector
        """
        self.parameters = parameters
        self.ctx = ctx
        self.faas = faas
        self.metrics = metrics
        self.now = now
        self.cluster = cluster
        self.replica_factory = replica_factory
        self.use_yield = use_yield
        if cluster and not replica_factory:
            raise AttributeError('Replica Factory must be set when given cluster.')

    def run(self):
        """
        Implements a latency-based scaling approach based on the HPA.

        In contrast to the official HPA, this implementation is not aggregating on a per-pod basis
        (i.e., average over all pods), but considers the available traces for each deployment over all traces.
        This means that we aggregate over all traces.


        The implementation considers all running Pods to be ready!

        In contrast to the official HPA, we cannot estimate the latency for Pods that have not yet received any requests.


        In the following the algorithm is described and parts of the documentation copied in case there are differences
        or notable changes to the official documentation.
        You can find the documentation here: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/

        "When scaling on CPU, if any pod has yet to become ready (it's still initializing, or possibly is unhealthy)
        or the most recent metric point for the pod was before it became ready, that pod is set aside as well."
        - the implementation always considers all available CPU metrics

        "If there were any missing metrics, the control plane recomputes the average more conservatively, assuming
        those pods were consuming 100% of the desired value in case of a scale down, and 0% in case of a scale up.
        This dampens the magnitude of any potential scale."
        - the implementation considers this

        "Furthermore, if any not-yet-ready pods were present, and the workload would have scaled up without
        factoring in missing metrics or not-yet-ready pods, the controller conservatively assumes that the
        not-yet-ready pods are consuming 0% of the desired metric, further dampening the magnitude of a scale up."
        - the implementation considers this

        "After factoring in the not-yet-ready pods and missing metrics, the controller recalculates the usage ratio.
        If the new ratio reverses the scale direction, or is within the tolerance, the controller doesn't take any
        scaling action. In other cases, the new ratio is used to decide any change to the number of Pods."
        - the implementation considers this
        """
        replica_service: FunctionReplicaService = self.ctx.replica_service
        deployment_service: FunctionDeploymentService = self.ctx.deployment_service
        trace_service: TraceService = self.ctx.trace_service

        for deployment in deployment_service.get_deployments():
            logger.info(f'HLPA scaling for function {deployment.name}')
            spec = self.parameters.get(deployment.name, None)
            if spec is None:
                continue
            if deployment.name == 'api-gateway':
                continue

            def access(r: ResponseRepresentation):
                return r.__dict__[spec.target_time_measure]

            running_pods = replica_service.get_function_replicas_of_deployment(deployment.name)
            # Setting to CONCEIVED good idea?
            pending_pods = replica_service.get_function_replicas_of_deployment(deployment.name, running=False,
                                                                               state=FunctionReplicaState.PENDING)
            conceiving_pods = replica_service.get_function_replicas_of_deployment(deployment.name, running=False,
                                                                                  state=FunctionReplicaState.CONCEIVED)

            if self.cluster is not None:
                in_cluster_running_pods = [x for x in running_pods if x.labels[zone_label] == self.cluster]
                no_in_cluster_running_pods = len(in_cluster_running_pods)
                in_cluster_pending_pods = [x for x in pending_pods if x.labels[zone_label] == self.cluster]
                no_in_cluster_pending_pods = len(in_cluster_pending_pods)
                in_cluster_conceiving_pods = [x for x in conceiving_pods if x.labels[zone_label] == self.cluster]
                no_in_cluster_conceiving_pods = len(in_cluster_conceiving_pods)
                no_in_cluster_pods = no_in_cluster_running_pods + no_in_cluster_pending_pods + no_in_cluster_conceiving_pods

            no_of_running_pods = len(running_pods) if running_pods is not None else 0
            no_of_pending_pods = len(pending_pods) if pending_pods is not None else 0
            no_of_conceiving_pods = len(conceiving_pods) if conceiving_pods is not None else 0
            logger.info(
                f'no of running pods: {no_of_running_pods}, no of pending pods: {no_of_pending_pods}, no of conceiving pods: {no_of_conceiving_pods}')
            no_of_pods = no_of_running_pods + no_of_pending_pods + no_of_conceiving_pods
            now = self.now()
            lookback_seconds_ago = now - spec.lookback
            if lookback_seconds_ago < 0:
                lookback_seconds_ago = 0
            logger.info(f"Fetch traces {spec.lookback} seconds ago")

            if self.cluster is not None:
                traces = trace_service.get_values_for_function_by_sent(deployment.name, lookback_seconds_ago, now,
                                                                       access,
                                                                       zone=self.cluster)
            else:
                traces = trace_service.get_values_for_function_by_sent(deployment.name, lookback_seconds_ago, now,
                                                                       access)
            if traces is None or len(traces) == 0:
                logger.info(f'No trace data for function: {deployment.name}, skip iteration')
                record = {
                    'fn': deployment.name,
                    'duration_agg': -1,
                    'target_duration': spec.target_duration,
                    'percentile': spec.percentile_duration,
                    'base_scale_ratio': -1,
                    'running_pods': no_in_cluster_running_pods,
                    'pending_pods': no_in_cluster_pending_pods,
                    'threshold_tolerance': spec.threshold_tolerance,
                }
                logger.info("Scale down because no traces were found")
                scale_down_containers = int(no_in_cluster_running_pods * 0.3)
                desired_replicas = no_in_cluster_running_pods - scale_down_containers
                if desired_replicas <= 0:
                    logger.info('Skip scale down, minimum number of pods running.')
                    continue
                logger.info(f'Number of desired  {desired_replicas}')
                logger.info(f'Number of all pods {no_in_cluster_running_pods}')
                logger.info(f'Number of replicas to be scaled downed {scale_down_containers}')
                # choose the last added containers
                # TODO: include pending pods, can lead to issues if too many pods
                #  are pending and not enoguh pods are running to remove
                all_pods = no_in_cluster_running_pods
                to_remove = in_cluster_running_pods[no_in_cluster_running_pods - scale_down_containers:]
                if len(to_remove) > 0:
                    if len(to_remove) > 20:
                        delete_length = len(to_remove) - 20
                        to_remove = to_remove[delete_length:]
                    self.scale_down(deployment.name, to_remove)
                if self.cluster:
                    record['cluster'] = self.cluster
                self.metrics.log('hlpa-decision', no_of_pods, **record)
                continue

            # percentile of rtt over all traces
            duration_agg = np.percentile(q=spec.percentile_duration, a=traces)
            mean = np.mean(a=traces)
            base_scale_ratio = duration_agg / spec.target_duration
            logger.info(
                f"Base scale ratio: {base_scale_ratio}."
                f" With a {spec.percentile_duration} percentile of {duration_agg} and a target of {spec.target_duration}, and a mean of {mean}")
            if 1 + spec.threshold_tolerance > base_scale_ratio > 1 - spec.threshold_tolerance:
                logger.info(
                    f'{spec.target_time_measure} percentile ({duration_agg}) was close enough to target ({spec.target_duration}) '
                    f'with a tolerance of {spec.threshold_tolerance}')
                record = {
                    'fn': deployment.name,
                    'duration_agg': duration_agg,
                    'target_duration': spec.target_duration,
                    'base_scale_ratio': base_scale_ratio,
                    'running_pods': no_of_running_pods,
                    'pending_pods': no_of_pending_pods,
                    'percentile': spec.percentile_duration,
                    'threshold_tolerance': spec.threshold_tolerance,
                }

                if self.cluster:
                    record['cluster'] = self.cluster
                self.metrics.log('hlpa-decision', no_of_pods, **record)
                continue

            desired_replicas = math.ceil(no_of_running_pods * (base_scale_ratio))
            logger.info(f"Desired {desired_replicas}")
            logger.info(f"Current Running {no_of_running_pods}")
            logger.info(f"Current Running + pending/conceived {no_of_pending_pods + no_of_conceiving_pods}")
            if desired_replicas == no_of_pods:
                logger.info(f"No scaling actions necessary for {deployment.name}")
                record = {
                    'fn': deployment.name,
                    'duration_agg': duration_agg,
                    'target_duration': spec.target_duration,
                    'base_scale_ratio': base_scale_ratio,
                    'running_pods': no_of_running_pods,
                    'pending_pods': no_of_pending_pods,
                    'percentile': spec.percentile_duration,
                    'threshold_tolerance': spec.threshold_tolerance,
                }

                if self.cluster:
                    record['cluster'] = self.cluster
                self.metrics.log('hlpa-decision', desired_replicas, **record)
                continue
            # logger.info(f"Scale to {desired_replicas} from {no_of_running_pods} without "
            #             f"factoring in not-yet-ready pods")
            logger.info(f"Scale to {desired_replicas} from {no_of_pods} with "
                        f"factoring in not-yet-ready pods")

            if desired_replicas > no_of_pods:
                logger.info("Scale up")
                # check if new number of pods is over the maximum. if yes => set to minimum
                scale_max = deployment.scaling_configuration.scale_max
                if scale_max is not None and desired_replicas > scale_max:
                    logger.info(
                        f'Number of desired replicas is bigger than scale max ({desired_replicas} > {scale_max}) -> scale to max replicas if possible.')
                    desired_replicas = scale_max

                scale_up_replicas_no = desired_replicas - no_of_pods
                logger.info(f'Number of desired  {desired_replicas}')
                logger.info(f'Number of all pods {no_of_pods}')
                logger.info(f'Number of replicas to be scaled up {scale_up_replicas_no}')
                scale_up_replicas = scale_up_replicas_no
                if self.cluster:
                    scale_up_replicas = []
                    for i in range(scale_up_replicas_no):
                        replica = self.replica_factory.create_replica(
                            {worker_role_label: 'true', 'origin_zone': self.cluster, zone_label: 'None',
                             'schedulerName': 'global-scheduler'},
                            deployment.deployment_ranking.get_first(), deployment)
                        scale_up_replicas.append(replica)

                if self.use_yield:
                    yield from self.faas.scale_up(deployment.name, scale_up_replicas)
                else:
                    self.faas.scale_up(deployment.name, scale_up_replicas)


            else:
                logger.info("Scale down")
                if desired_replicas == 0 and no_of_pods == 1 and no_in_cluster_pods == 1:
                    desired_replicas = 1
                scale_down_containers = no_in_cluster_pods - desired_replicas
                if scale_down_containers < 0:
                    scale_down_containers = no_in_cluster_pods - 1
                logger.info(f'Number of desired  {desired_replicas}')
                logger.info(f'Number of all pods in cluster {no_in_cluster_pods}')
                logger.info(f'Number of replicas to be scaled downed {scale_down_containers}')
                # choose the last added containers
                # TODO: include pending pods, can lead to issues if too many pods
                #  are pending and not enoguh pods are running to remove
                all_pods = in_cluster_running_pods + in_cluster_pending_pods + in_cluster_conceiving_pods
                to_remove = []
                not_in_cluster = []
                for pod in all_pods:
                    if pod.labels[zone_label] == self.cluster:
                        if len(to_remove) == scale_down_containers:
                            break
                        to_remove.append(pod)
                    else:
                        not_in_cluster.append(pod)

                # if len(to_remove) < scale_down_containers:
                #     missing = scale_down_containers - len(to_remove)
                #     to_remove.extend(not_in_cluster[:missing])
                if len(to_remove) > 0:
                    if len(to_remove) > 20:
                        delete_length = len(to_remove) - 20
                        to_remove = to_remove[delete_length:]
                    if self.use_yield:
                        yield from self.scale_down(deployment.name, to_remove)
                    else:
                        self.scale_down(deployment.name, to_remove)

            record = {
                's': self.now(),
                'fn': deployment.name,
                'duration_agg': duration_agg,
                'target_duration': spec.target_duration,
                'base_scale_ratio': base_scale_ratio,
                'running_pods': no_of_running_pods,
                'pending_pods': no_of_pending_pods,
                'threshold_tolerance': spec.threshold_tolerance,
                'percentile': spec.percentile_duration
            }

            if self.cluster:
                record['cluster'] = self.cluster
            self.metrics.log('hlpa-decision', desired_replicas, **record)

    def scale_down(self, function: str, remove: Union[int, List[FunctionReplica]]):
        self.faas.scale_down(function, remove)

    def scale_up(self, function: str, add: Union[int, List[FunctionReplica]]):
        self.faas.scale_up(function, add)
