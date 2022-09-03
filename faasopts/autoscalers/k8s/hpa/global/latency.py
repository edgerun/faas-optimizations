import datetime
import logging
import math
from dataclasses import dataclass
from typing import Dict, Callable

import numpy as np
from faas.context import PlatformContext, FunctionReplicaService, FunctionDeploymentService, TraceService
from faas.system import FaasSystem, Metrics, FunctionReplicaState, Clock

from faasopts.autoscalers.api import BaseAutoscaler

logger = logging.getLogger(__name__)


@dataclass
class HorizontalLatencyPodAutoscalerParameters:
    # the past (in seconds) that should be considered when looking at monitoring data
    lookback: int

    # either latency (in ms) or rtt (in s)
    target_time_measure: str

    #  the tolerance within the target metric can be without triggering in our out scaling
    threshold_tolerance: float = 0.1
    # ms when latency, s when rtt
    target_duration: float = 100
    # which percentile should be used to calculate ratio
    percentile_duration: float = 90


class HorizontalLatencyPodAutoscaler(BaseAutoscaler):
    """
    This Optimizer implementation is based on the official default Kubernetes HPA and uses latency to determine
    the number of replicas.

    Reference: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
    """

    def __init__(self, parameters: Dict[str, HorizontalLatencyPodAutoscalerParameters], ctx: PlatformContext,
                 faas: FaasSystem,
                 metrics: Metrics, now: Callable[[], float]):
        """
        Initializes the HPA latency-based implementation
        :param parameters: HPA parameters per deployment that dictate various configuration values
        :param ctx: the HPA will get any information (i.e., monitoring data) from the context
        :param faas: the system will be invoked when scaling in our out
        :param metrics: is used to log any scaling decisions for later analysis
        """
        self.parameters = parameters
        self.ctx = ctx
        self.faas = faas
        self.metrics = metrics
        self.now = now

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
            logger.info(f'HLPA scaling for function {deployment.fn_name}')
            spec = self.parameters.get(deployment.fn_name, None)
            if spec is None:
                continue
            running_pods = replica_service.get_function_replicas_of_deployment(deployment.original_name)
            pending_pods = replica_service.get_function_replicas_of_deployment(deployment.original_name, running=False,
                                                                               state=FunctionReplicaState.PENDING)
            no_of_running_pods = len(running_pods)
            no_of_pending_pods = len(pending_pods)
            no_of_pods = no_of_running_pods + no_of_pending_pods
            now = self.now()
            lookback_seconds_ago = now - spec.lookback
            logger.info(f"Fetch traces {spec.lookback} seconds ago")
            traces = trace_service.get_traces_for_function(deployment, lookback_seconds_ago, now)
            if len(traces) == 0:
                logger.info(f'No trace data for function: {deployment.fn_name}, skip iteration')
                record = {
                    'fn': deployment.fn_name,
                    'duration_agg': -1,
                    'target_duration': spec.target_duration,
                    'percentile': spec.percentile_duration,
                    'base_scale_ratio': -1,
                    'running_pods': no_of_running_pods,
                    'pending_pods': no_of_pending_pods,
                    'threshold_tolerance': spec.threshold_tolerance,
                }
                self.metrics.log('hlpa-decision', no_of_pods, **record)
                continue

            if spec.target_time_measure == 'rtt':
                traces[spec.target_time_measure] = traces[spec.target_time_measure] * 1000

            # percentile of rtt over all traces
            duration_agg = np.percentile(q=spec.percentile_duration, a=traces[spec.target_time_measure])
            mean = np.mean(a=traces[spec.target_time_measure])
            base_scale_ratio = duration_agg / spec.target_duration
            logger.info(
                f"Base scale ratio: {base_scale_ratio}."
                f" With a {spec.percentile_duration} percentile of {duration_agg} and a target of {spec.target_duration}, and a mean of {mean}")
            if 1 + spec.threshold_tolerance > base_scale_ratio > 1 - spec.threshold_tolerance:
                logger.info(
                    f'{spec.target_time_measure} percentile ({duration_agg}) was close enough to target ({spec.target_duration}) '
                    f'with a tolerance of {spec.threshold_tolerance}')
                record = {
                    'fn': deployment.fn_name,
                    'duration_agg': duration_agg,
                    'target_duration': spec.target_duration,
                    'base_scale_ratio': base_scale_ratio,
                    'running_pods': no_of_running_pods,
                    'pending_pods': no_of_pending_pods,
                    'percentile': spec.percentile_duration,
                    'threshold_tolerance': spec.threshold_tolerance,
                }
                self.metrics.log('hlpa-decision', no_of_pods, **record)
                continue

            desired_replicas = math.ceil(no_of_running_pods * (base_scale_ratio))
            if desired_replicas == no_of_running_pods:
                logger.info(f"No scaling actions necessary for {deployment.fn_name}")
                record = {
                    'fn': deployment.fn_name,
                    'duration_agg': duration_agg,
                    'target_duration': spec.target_duration,
                    'base_scale_ratio': base_scale_ratio,
                    'running_pods': no_of_running_pods,
                    'pending_pods': no_of_pending_pods,
                    'percentile': spec.percentile_duration,
                    'threshold_tolerance': spec.threshold_tolerance,
                }
                self.metrics.log('hlpa-decision', desired_replicas, **record)
                continue
            logger.info(f"Scale to {desired_replicas} from {no_of_running_pods} without "
                        f"factoring in not-yet-ready pods")
            if desired_replicas > no_of_running_pods:
                logger.info("Scale up")

                # check if new number of pods is over the maximum. if yes => set to minimum
                scale_max = deployment.scaling_configuration.scale_max
                if scale_max is not None and desired_replicas > scale_max:
                    desired_replicas = scale_max

                scale_up_containers = desired_replicas - no_of_pods
                self.faas.scale_up(deployment.name, scale_up_containers)


            else:
                logger.info("Scale down")

                scale_down_containers = no_of_pods - desired_replicas

                # choose the last added containers
                # TODO: include pending pods, can lead to issues if too many pods
                #  are pending and not enoguh pods are running to remove
                to_remove = running_pods[no_of_running_pods - scale_down_containers:]
                self.faas.scale_down(deployment.name, to_remove)

            record = {
                's': self.now(),
                'fn': deployment.fn_name,
                'duration_agg': duration_agg,
                'target_duration': spec.target_duration,
                'base_scale_ratio': base_scale_ratio,
                'running_pods': no_of_running_pods,
                'pending_pods': no_of_pending_pods,
                'threshold_tolerance': spec.threshold_tolerance,
                'percentile': spec.percentile_duration
            }
            self.metrics.log('hlpa-decision', desired_replicas, **record)
