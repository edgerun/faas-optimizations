import logging
import logging
import math
import threading
from dataclasses import dataclass
from typing import Dict, Callable, Union, List

import numpy as np
from dataclasses_json import dataclass_json
from faas.context import PlatformContext, FunctionReplicaService, FunctionDeploymentService, TraceService, \
    ResponseRepresentation, FunctionReplicaFactory
from faas.system import FaasSystem, Metrics, FunctionReplicaState, FunctionReplica
from faas.util.constant import zone_label, worker_role_label

from faasopts.autoscalers.api import BaseAutoscaler
from faasopts.autoscalers.base.hpa.decentralized.latency import DecentralizedHorizontalLatencyPodAutoscaler

logger = logging.getLogger(__name__)


@dataclass_json
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


class K8sDecentralizedHorizontalLatencyPodAutoscaler(DecentralizedHorizontalLatencyPodAutoscaler):
    """
    This Optimizer implementation is based on the official default Kubernetes HPA and uses latency to determine
    the number of replicas.

    Reference: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
    """

    def __init__(self, parameters: Dict[str, HorizontalLatencyPodAutoscalerParameters], ctx: PlatformContext,
                 faas: FaasSystem,
                 metrics: Metrics, now: Callable[[], float], cluster: str = None,
                 replica_factory: FunctionReplicaFactory = None):
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
        self.lock = threading.Lock()
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
        actions = super().run()
        for function, replicas, scale_up in actions:
            if scale_up:
                self.scale_up(function, replicas)
            else:
                self.scale_down(function, replicas)

    def scale_down(self, function: str, remove: Union[int, List[FunctionReplica]]):
        self.faas.scale_down(function, remove)

    def scale_up(self, function: str, add: Union[int, List[FunctionReplica]]):
        self.faas.scale_up(function, add)
