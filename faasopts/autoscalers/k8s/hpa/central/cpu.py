import logging
import logging
import math
from dataclasses import dataclass
from typing import Dict, Callable, List, Union

import numpy as np
import pandas as pd
from faas.context import PlatformContext, FunctionReplicaService, FunctionDeploymentService, TelemetryService
from faas.system import FaasSystem, Metrics, FunctionReplicaState, FunctionDeployment, FunctionReplica

from faasopts.autoscalers.api import BaseAutoscaler

logger = logging.getLogger(__name__)


@dataclass
class HorizontalCpuReplicaAutoscalerParameters:
    # this value indicates which column from the dataframe of the telemetry service should be used
    # to scale
    cpu_column: str

    # the past (in seconds) that should be considered when looking at monitoring data
    lookback: int

    #  the tolerance within the target metric can be without triggering in our out scaling
    threshold_tolerance: float = 0.1

    # default target utilization is 80%
    # ref: https://github.com/kubernetes/community/blob/master/contributors/design-proposals/autoscaling/horizontal-replica-autoscaler.md
    target_avg_utilization: float = 80


class HorizontalCpuReplicaAutoscaler(BaseAutoscaler):
    """
    This Optimizer implementation is based on the official default Kubernetes HPA and uses CPU  usage to determine
    the number of replicas.

    Reference: https://kubernetes.io/docs/tasks/run-application/horizontal-replica-autoscale/
    """

    def __init__(self, parameters: Dict[str, HorizontalCpuReplicaAutoscalerParameters], ctx: PlatformContext,
                 faas: FaasSystem, metrics: Metrics, now: Callable[[], float]):
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

         The implementation considers all running replicas to be ready!

         In the following the algorithm is described and parts of the documentation copied in case there are differences
         or notable changes to the official documentation.
         You can find the documentation here: https://kubernetes.io/docs/tasks/run-application/horizontal-replica-autoscale/

         "When scaling on CPU, if any replica has yet to become ready (it's still initializing, or possibly is unhealthy)
         or the most recent metric point for the replica was before it became ready, that replica is set aside as well."
         - the implementation always considers all available CPU metrics

         "If there were any missing metrics, the control plane recomputes the average more conservatively, assuming
         those replicas were consuming 100% of the desired value in case of a scale down, and 0% in case of a scale up.
         This dampens the magnitude of any potential scale."
         - the implementation considers this

         "Furthermore, if any not-yet-ready replicas were present, and the workload would have scaled up without
         factoring in missing metrics or not-yet-ready replicas, the controller conservatively assumes that the
         not-yet-ready replicas are consuming 0% of the desired metric, further dampening the magnitude of a scale up."
         - the implementation considers this

         "After factoring in the not-yet-ready replicas and missing metrics, the controller recalculates the usage ratio.
         If the new ratio reverses the scale direction, or is within the tolerance, the controller doesn't take any
         scaling action. In other cases, the new ratio is used to decide any change to the number of replicas."
         - the implementation considers this
        """
        replica_service: FunctionReplicaService = self.ctx.replica_service
        deployment_service: FunctionDeploymentService = self.ctx.deployment_service
        telemetry_service: TelemetryService = self.ctx.telemetry_service
        deployments: List[FunctionDeployment] = deployment_service.get_deployments()
        for deployment in deployments:
            logger.info(f'HPA scaling for function {deployment.name}')
            spec = self.parameters.get(deployment.name, None)
            if spec is None:
                continue
            cpu_column = spec.cpu_column
            cpu_usages = []
            running_replica = replica_service.get_function_replicas_of_deployment(deployment.name)
            pending_replica = replica_service.get_function_replicas_of_deployment(deployment.name, running=False,
                                                                                  state=FunctionReplicaState.PENDING)
            no_of_running_replicas = len(running_replica)
            no_of_pending_replicas = len(pending_replica)
            no_of_replicas = no_of_running_replicas + no_of_pending_replicas

            # in case of latency, we don't consider missing cpu
            # we just check if traces are here or not

            missing_cpu = 0
            for replica in running_replica:
                now = self.now()
                lookback_seconds_ago = now - spec.lookback
                cpu = telemetry_service.get_replica_cpu(replica.replica_id, lookback_seconds_ago, now)
                if cpu is None or len(cpu) == 0:
                    missing_cpu += 1
                else:
                    cpu_usages.append(cpu)
            if len(cpu_usages) > 0:
                df_running = pd.concat(cpu_usages)
            else:
                logger.info(f'No CPU usage data for function: {deployment.name}, skip iteration')
                record = {
                    'fn': deployment.name,
                    'cpu_mean': -1,
                    'recalculated_cpu_mean': -1,
                    'target_utilization': spec.target_avg_utilization,
                    'base_scale_ratio': -1,
                    'running_replica': no_of_running_replicas,
                    'pending_replica': no_of_pending_replicas,
                    'missing_cpu': missing_cpu,
                    'threshold_tolerance': spec.threshold_tolerance,
                }
                self.metrics.log('hcpa-decision', no_of_replicas, **record)
                continue

            # mean utilization over all running replicas
            mean_per_replica = df_running.groupby('replica_id').mean()

            cpu_mean = mean_per_replica[cpu_column].mean()

            base_scale_ratio = cpu_mean / spec.target_avg_utilization
            logger.info(
                f"Base scale ratio without factoring in missing cpu or not-yet-ready replicas is: {base_scale_ratio}."
                f" With a mean cpu of {cpu_mean} and a target of {spec.target_avg_utilization}")
            if 1 + spec.threshold_tolerance > base_scale_ratio > 1 - spec.threshold_tolerance:
                logger.info(f'Utilization ({cpu_mean}) was close enough to target ({spec.target_avg_utilization}) '
                            f'with a tolerance of {spec.threshold_tolerance}')
                record = {
                    'fn': deployment.name,
                    'cpu_mean': cpu_mean,
                    'recalculated_cpu_mean': -1,
                    'target_utilization': spec.target_avg_utilization,
                    'base_scale_ratio': base_scale_ratio,
                    'running_replica': no_of_running_replicas,
                    'pending_replica': no_of_pending_replicas,
                    'missing_cpu': missing_cpu,
                    'threshold_tolerance': spec.threshold_tolerance,
                }
                self.metrics.log('hcpa-decision', no_of_replicas, **record)
                continue

            desired_replicas = math.ceil(no_of_running_replicas * (base_scale_ratio))
            if desired_replicas == no_of_running_replicas:
                logger.info(f"No scaling actions necessary for {deployment.name}")
                record = {
                    'fn': deployment.name,
                    'cpu_mean': cpu_mean,
                    'recalculated_cpu_mean': -1,
                    'target_utilization': spec.target_avg_utilization,
                    'base_scale_ratio': base_scale_ratio,
                    'running_replica': no_of_running_replicas,
                    'pending_replica': no_of_pending_replicas,
                    'missing_cpu': missing_cpu,
                    'threshold_tolerance': spec.threshold_tolerance,
                }
                self.metrics.log('hcpa-decision', no_of_replicas, **record)
                continue

            logger.info(f"Scale to {desired_replicas} from {no_of_running_replicas} without "
                        f"factoring in not-yet-ready replicas")
            if desired_replicas > no_of_running_replicas:
                logger.info("Scale up")
                recalculated_cpu = mean_per_replica[cpu_column].tolist()

                # add 0% usage for all missing cpu replicas
                for i in range(0, missing_cpu):
                    recalculated_cpu.append(0)

                # add 0% usage for all pending (not-yet-ready) replicas
                for i in range(0, len(pending_replica)):
                    recalculated_cpu.append(0)

                # recalculate cpu mean
                recalculated_cpu_mean = np.mean(recalculated_cpu)

                # recalculate ratio
                recalculated_ratio = recalculated_cpu_mean / spec.target_avg_utilization

                logger.info(
                    f"Recalculated scale ratio without factoring in missing cpu or not-yet-ready replicas is: "
                    f"{recalculated_ratio}."
                    f" With a mean cpu of {recalculated_cpu_mean} and a target of {spec.target_avg_utilization}")

                # recalculate desired replicas
                recalculated_desired_replicas = math.ceil(no_of_replicas * recalculated_ratio)

                # check if ratio is now within tolerance
                if 1 + spec.threshold_tolerance > recalculated_ratio > 1 - spec.threshold_tolerance:
                    logger.info(f'Recalculated utilization ({recalculated_cpu_mean}) was close enough to target '
                                f'({spec.target_avg_utilization}) '
                                f'with a tolerance of {spec.threshold_tolerance}')
                    record = {
                        'fn': deployment.name,
                        'cpu_mean': cpu_mean,
                        'recalculated_cpu_mean': -1,
                        'target_utilization': spec.target_avg_utilization,
                        'base_scale_ratio': base_scale_ratio,
                        'running_replica': no_of_running_replicas,
                        'pending_replica': no_of_pending_replicas,
                        'missing_cpu': missing_cpu,
                        'threshold_tolerance': spec.threshold_tolerance,
                    }
                    self.metrics.log('hcpa-decision', recalculated_desired_replicas, **record)
                    continue

                # check if action is now reveresed
                if recalculated_desired_replicas <= no_of_replicas:
                    logger.info("Recalculation reversed scaling action and therefore no action happens")
                    record = {
                        'fn': deployment.name,
                        'cpu_mean': cpu_mean,
                        'recalculated_cpu_mean': -1,
                        'target_utilization': spec.target_avg_utilization,
                        'base_scale_ratio': base_scale_ratio,
                        'running_replica': no_of_running_replicas,
                        'pending_replica': no_of_pending_replicas,
                        'missing_cpu': missing_cpu,
                        'threshold_tolerance': spec.threshold_tolerance,
                    }
                    self.metrics.log('hcpa-decision', recalculated_desired_replicas, **record)
                    continue

                # check if new number of replicas is over the maximum. if yes => set to minimum
                scale_max = deployment.scaling_configuration.scale_max
                if scale_max is not None and recalculated_desired_replicas > scale_max:
                    recalculated_desired_replicas = scale_max

                scale_up_containers = recalculated_desired_replicas - no_of_replicas
                yield from self.scale_up(deployment.name, scale_up_containers)

            else:
                logger.info("Scale down")
                recalculated_cpu = mean_per_replica[cpu_column].tolist()
                no_of_replicas = no_of_running_replicas

                # add 100% usage for all missing cpu replicas
                for i in range(0, missing_cpu):
                    recalculated_cpu.append(100)
                    no_of_replicas += 1

                # recalculate cpu mean
                recalculated_cpu_mean = np.mean(recalculated_cpu)
                if type(recalculated_cpu_mean) is np.ndarray:
                    recalculated_cpu_mean = recalculated_cpu_mean[0]

                # recalculate ratio
                recalculated_ratio = recalculated_cpu_mean / spec.target_avg_utilization

                logger.info(
                    f"Recalculated scale ratio without factoring in missing cpu or not-yet-ready replicas is: "
                    f"{recalculated_ratio}."
                    f" With a mean cpu of {recalculated_cpu_mean} and a target of {spec.target_avg_utilization}")

                # recalculate desired replicas
                recalculated_desired_replicas = math.ceil(no_of_replicas * recalculated_ratio)

                # check if ratio is now within tolerance
                if 1 + spec.threshold_tolerance > recalculated_ratio > 1 - spec.threshold_tolerance:
                    logger.info(
                        f'Recalculated utilization ({recalculated_cpu_mean}) was close enough '
                        f'to target ({spec.target_avg_utilization}) '
                        f'with a tolerance of {spec.threshold_tolerance}')
                    record = {
                        'fn': deployment.name,
                        'cpu_mean': cpu_mean,
                        'recalculated_cpu_mean': -1,
                        'target_utilization': spec.target_avg_utilization,
                        'base_scale_ratio': base_scale_ratio,
                        'running_replica': no_of_running_replicas,
                        'pending_replica': no_of_pending_replicas,
                        'missing_cpu': missing_cpu,
                        'threshold_tolerance': spec.threshold_tolerance,
                    }
                    self.metrics.log('hcpa-decision', recalculated_desired_replicas, **record)
                    continue

                # check if action is now reveresed
                if recalculated_desired_replicas >= no_of_replicas:
                    logger.info("Recalculation reversed scaling action and therefore no action happens")
                    record = {
                        'fn': deployment.name,
                        'cpu_mean': cpu_mean,
                        'recalculated_cpu_mean': -1,
                        'target_utilization': spec.target_avg_utilization,
                        'base_scale_ratio': base_scale_ratio,
                        'running_replica': no_of_running_replicas,
                        'pending_replica': no_of_pending_replicas,
                        'missing_cpu': missing_cpu,
                        'threshold_tolerance': spec.threshold_tolerance,
                    }
                    self.metrics.log('hcpa-decision', recalculated_desired_replicas, **record)
                    continue

                # check if new number of replicas is below the minimum. if yes => set to minimum
                scale_min = deployment.scaling_configuration.scale_min
                if recalculated_desired_replicas < scale_min:
                    recalculated_desired_replicas = scale_min

                scale_down_containers = no_of_replicas - recalculated_desired_replicas

                # choose the last added containers
                to_remove = running_replica[no_of_running_replicas - scale_down_containers:]
                if len(to_remove) > 0:
                    self.scale_down(deployment.fn.name, to_remove)

            record = {
                'fn': deployment.name,
                'cpu_mean': cpu_mean,
                'recalculated_cpu_mean': recalculated_cpu_mean,
                'target_utilization': spec.target_avg_utilization,
                'base_scale_ratio': base_scale_ratio,
                'running_replica': no_of_running_replicas,
                'pending_replica': no_of_pending_replicas,
                'missing_cpu': missing_cpu,
                'threshold_tolerance': spec.threshold_tolerance,
            }
            self.metrics.log('hcpa-decision', recalculated_desired_replicas, **record)

    def stop(self):
        pass

    def scale_down(self, function: str, remove: Union[int, List[FunctionReplica]]):
        self.faas.scale_down(function, remove)

    def scale_up(self, function: str, add: Union[int, List[FunctionReplica]]):
        self.faas.scale_up(function, add)
