import datetime
import logging
import math
from dataclasses import dataclass
from typing import Dict, Callable

import numpy as np
import pandas as pd
from faas.context import PlatformContext, FunctionReplicaService, FunctionDeploymentService, TelemetryService
from faas.system import FaasSystem, Metrics, FunctionReplicaState, Clock

from faasopts.autoscalers.api import BaseAutoscaler

logger = logging.getLogger(__name__)


@dataclass
class HorizontalCpuPodAutoscalerParameters:
    # this value indicates which column from the dataframe of the telemetry service should be used
    # to scale
    cpu_column: str

    # the past (in seconds) that should be considered when looking at monitoring data
    lookback: int

    #  the tolerance within the target metric can be without triggering in our out scaling
    threshold_tolerance: float = 0.1

    # default target utilization is 80%
    # ref: https://github.com/kubernetes/community/blob/master/contributors/design-proposals/autoscaling/horizontal-pod-autoscaler.md
    target_avg_utilization: float = 80




class HorizontalCpuPodAutoscaler(BaseAutoscaler):
    """
    This Optimizer implementation is based on the official default Kubernetes HPA and uses CPU  usage to determine
    the number of Pods.

    Reference: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
    """

    def __init__(self, parameters: Dict[str, HorizontalCpuPodAutoscalerParameters], ctx: PlatformContext, faas: FaasSystem,
                 metrics: Metrics, now: Callable[[], float]):
        """
        Initializes the HPA CPU-based implementation
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

    def setup(self):
        pass

    def run(self):
        """
         Implements a customizable scaling approach based on the HPA.

         The implementation considers all running Pods to be ready!

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
        telemetry_service: TelemetryService = self.ctx.telemetry_service
        for deployment in deployment_service.get_deployments():
            logger.info(f'HPA scaling for function {deployment.fn_name}')
            spec = self.parameters.get(deployment.fn_name, None)
            if spec is None:
                continue
            cpu_column = spec.cpu_column
            cpu_usages = []
            running_pods = replica_service.get_function_replicas_of_deployment(deployment.original_name)
            pending_pods = replica_service.get_function_replicas_of_deployment(deployment.original_name, running=False,
                                                                               state=FunctionReplicaState.PENDING)
            no_of_running_pods = len(running_pods)
            no_of_pending_pods = len(pending_pods)
            no_of_pods = no_of_running_pods + no_of_pending_pods

            # in case of latency, we don't consider missing cpu
            # we just check if traces are here or not

            missing_cpu = 0
            for pod in running_pods:
                now = self.now()
                lookback_seconds_ago = now - spec.lookback
                cpu = telemetry_service.get_replica_cpu(pod.container_id, lookback_seconds_ago, now)
                if cpu is None or len(cpu) == 0:
                    missing_cpu += 1
                else:
                    cpu_usages.append(cpu)
            if len(cpu_usages) > 0:
                df_running = pd.concat(cpu_usages)
            else:
                logger.info(f'No CPU usage data for function: {deployment.fn_name}, skip iteration')
                record = {
                    'fn': deployment.fn_name,
                    'cpu_mean': -1,
                    'recalculated_cpu_mean': -1,
                    'target_utilization': spec.target_avg_utilization,
                    'base_scale_ratio': -1,
                    'running_pods': no_of_running_pods,
                    'pending_pods': no_of_pending_pods,
                    'missing_cpu': missing_cpu,
                    'threshold_tolerance': spec.threshold_tolerance,
                }
                self.metrics.log('hcpa-decision', no_of_pods, **record)
                continue

            # mean utilization over all running pods
            mean_per_pod = df_running.groupby('replica_id').mean()

            cpu_mean = mean_per_pod[cpu_column].mean()

            base_scale_ratio = cpu_mean / spec.target_avg_utilization
            logger.info(
                f"Base scale ratio without factoring in missing cpu or not-yet-ready pods is: {base_scale_ratio}."
                f" With a mean cpu of {cpu_mean} and a target of {spec.target_avg_utilization}")
            if 1 + spec.threshold_tolerance > base_scale_ratio > 1 - spec.threshold_tolerance:
                logger.info(f'Utilization ({cpu_mean}) was close enough to target ({spec.target_avg_utilization}) '
                            f'with a tolerance of {spec.threshold_tolerance}')
                record = {
                    'fn': deployment.fn_name,
                    'cpu_mean': cpu_mean,
                    'recalculated_cpu_mean': -1,
                    'target_utilization': spec.target_avg_utilization,
                    'base_scale_ratio': base_scale_ratio,
                    'running_pods': no_of_running_pods,
                    'pending_pods': no_of_pending_pods,
                    'missing_cpu': missing_cpu,
                    'threshold_tolerance': spec.threshold_tolerance,
                }
                self.metrics.log('hcpa-decision', no_of_pods, **record)
                continue

            desired_replicas = math.ceil(no_of_running_pods * (base_scale_ratio))
            if desired_replicas == no_of_running_pods:
                logger.info(f"No scaling actions necessary for {deployment.fn_name}")
                record = {
                    'fn': deployment.fn_name,
                    'cpu_mean': cpu_mean,
                    'recalculated_cpu_mean': -1,
                    'target_utilization': spec.target_avg_utilization,
                    'base_scale_ratio': base_scale_ratio,
                    'running_pods': no_of_running_pods,
                    'pending_pods': no_of_pending_pods,
                    'missing_cpu': missing_cpu,
                    'threshold_tolerance': spec.threshold_tolerance,
                }
                self.metrics.log('hcpa-decision', no_of_pods, **record)
                continue

            logger.info(f"Scale to {desired_replicas} from {no_of_running_pods} without "
                        f"factoring in not-yet-ready pods")
            if desired_replicas > no_of_running_pods:
                logger.info("Scale up")
                recalculated_cpu = mean_per_pod[cpu_column].tolist()

                # add 0% usage for all missing cpu pods
                for i in range(0, missing_cpu):
                    recalculated_cpu.append(0)

                # add 0% usage for all pending (not-yet-ready) pods
                for i in range(0, len(pending_pods)):
                    recalculated_cpu.append(0)

                # recalculate cpu mean
                recalculated_cpu_mean = np.mean(recalculated_cpu)

                # recalculate ratio
                recalculated_ratio = recalculated_cpu_mean / spec.target_avg_utilization

                logger.info(
                    f"Recalculated scale ratio without factoring in missing cpu or not-yet-ready pods is: "
                    f"{recalculated_ratio}."
                    f" With a mean cpu of {recalculated_cpu_mean} and a target of {spec.target_avg_utilization}")

                # recalculate desired replicas
                recalculated_desired_replicas = math.ceil(no_of_pods * recalculated_ratio)

                # check if ratio is now within tolerance
                if 1 + spec.threshold_tolerance > recalculated_ratio > 1 - spec.threshold_tolerance:
                    logger.info(f'Recalculated utilization ({recalculated_cpu_mean}) was close enough to target '
                                f'({spec.target_avg_utilization}) '
                                f'with a tolerance of {spec.threshold_tolerance}')
                    record = {
                        'fn': deployment.fn_name,
                        'cpu_mean': cpu_mean,
                        'recalculated_cpu_mean': -1,
                        'target_utilization': spec.target_avg_utilization,
                        'base_scale_ratio': base_scale_ratio,
                        'running_pods': no_of_running_pods,
                        'pending_pods': no_of_pending_pods,
                        'missing_cpu': missing_cpu,
                        'threshold_tolerance': spec.threshold_tolerance,
                    }
                    self.metrics.log('hcpa-decision', recalculated_desired_replicas, **record)
                    continue

                # check if action is now reveresed
                if recalculated_desired_replicas <= no_of_pods:
                    logger.info("Recalculation reversed scaling action and therefore no action happens")
                    record = {
                        'fn': deployment.fn_name,
                        'cpu_mean': cpu_mean,
                        'recalculated_cpu_mean': -1,
                        'target_utilization': spec.target_avg_utilization,
                        'base_scale_ratio': base_scale_ratio,
                        'running_pods': no_of_running_pods,
                        'pending_pods': no_of_pending_pods,
                        'missing_cpu': missing_cpu,
                        'threshold_tolerance': spec.threshold_tolerance,
                    }
                    self.metrics.log('hcpa-decision', recalculated_desired_replicas, **record)
                    continue

                # check if new number of pods is over the maximum. if yes => set to minimum
                scale_max = deployment.scaling_configuration.scale_max
                if scale_max is not None and recalculated_desired_replicas > scale_max:
                    recalculated_desired_replicas = scale_max

                scale_up_containers = recalculated_desired_replicas - no_of_pods

                self.faas.scale_up(deployment.name, scale_up_containers)

            else:
                logger.info("Scale down")
                recalculated_cpu = mean_per_pod[cpu_column].tolist()
                no_of_pods = no_of_running_pods

                # add 100% usage for all missing cpu pods
                for i in range(0, missing_cpu):
                    recalculated_cpu.append(100)
                    no_of_pods += 1

                # recalculate cpu mean
                recalculated_cpu_mean = np.mean(recalculated_cpu)
                if type(recalculated_cpu_mean) is np.ndarray:
                    recalculated_cpu_mean = recalculated_cpu_mean[0]

                # recalculate ratio
                recalculated_ratio = recalculated_cpu_mean / spec.target_avg_utilization

                logger.info(
                    f"Recalculated scale ratio without factoring in missing cpu or not-yet-ready pods is: "
                    f"{recalculated_ratio}."
                    f" With a mean cpu of {recalculated_cpu_mean} and a target of {spec.target_avg_utilization}")

                # recalculate desired replicas
                recalculated_desired_replicas = math.ceil(no_of_pods * recalculated_ratio)

                # check if ratio is now within tolerance
                if 1 + spec.threshold_tolerance > recalculated_ratio > 1 - spec.threshold_tolerance:
                    logger.info(
                        f'Recalculated utilization ({recalculated_cpu_mean}) was close enough '
                        f'to target ({spec.target_avg_utilization}) '
                        f'with a tolerance of {spec.threshold_tolerance}')
                    record = {
                        'fn': deployment.fn_name,
                        'cpu_mean': cpu_mean,
                        'recalculated_cpu_mean': -1,
                        'target_utilization': spec.target_avg_utilization,
                        'base_scale_ratio': base_scale_ratio,
                        'running_pods': no_of_running_pods,
                        'pending_pods': no_of_pending_pods,
                        'missing_cpu': missing_cpu,
                        'threshold_tolerance': spec.threshold_tolerance,
                    }
                    self.metrics.log('hcpa-decision', recalculated_desired_replicas, **record)
                    continue

                # check if action is now reveresed
                if recalculated_desired_replicas >= no_of_pods:
                    logger.info("Recalculation reversed scaling action and therefore no action happens")
                    record = {
                        'fn': deployment.fn_name,
                        'cpu_mean': cpu_mean,
                        'recalculated_cpu_mean': -1,
                        'target_utilization': spec.target_avg_utilization,
                        'base_scale_ratio': base_scale_ratio,
                        'running_pods': no_of_running_pods,
                        'pending_pods': no_of_pending_pods,
                        'missing_cpu': missing_cpu,
                        'threshold_tolerance': spec.threshold_tolerance,
                    }
                    self.metrics.log('hcpa-decision', recalculated_desired_replicas, **record)
                    continue

                # check if new number of pods is below the minimum. if yes => set to minimum
                scale_min = deployment.scaling_configuration.scale_min
                if recalculated_desired_replicas < scale_min:
                    recalculated_desired_replicas = scale_min

                scale_down_containers = no_of_pods - recalculated_desired_replicas

                # choose the last added containers
                to_remove = running_pods[no_of_running_pods - scale_down_containers:]
                self.faas.scale_down(deployment.name, to_remove)

            record = {
                'fn': deployment.fn_name,
                'cpu_mean': cpu_mean,
                'recalculated_cpu_mean': recalculated_cpu_mean,
                'target_utilization': spec.target_avg_utilization,
                'base_scale_ratio': base_scale_ratio,
                'running_pods': no_of_running_pods,
                'pending_pods': no_of_pending_pods,
                'missing_cpu': missing_cpu,
                'threshold_tolerance': spec.threshold_tolerance,
            }
            self.metrics.log('hcpa-decision', recalculated_desired_replicas, **record)

    def stop(self):
        pass
