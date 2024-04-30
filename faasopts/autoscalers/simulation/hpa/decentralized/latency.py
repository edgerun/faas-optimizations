import logging
from typing import Union, List

from faas.system import FunctionReplica

from faasopts.autoscalers.base.hpa.decentralized.latency import DecentralizedHorizontalLatencyPodAutoscaler

logger = logging.getLogger(__name__)


class SimulationDecentralizedHorizontalLatencyPodAutoscaler(DecentralizedHorizontalLatencyPodAutoscaler):
    """
    This Optimizer implementation is based on the official default Kubernetes HPA and uses latency to determine
    the number of replicas.

    Reference: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
    """

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
                yield from self.scale_up(function, replicas)
            else:
                yield from self.scale_down(function, replicas)

    def scale_down(self, function: str, remove: Union[int, List[FunctionReplica]]):
        yield from self.faas.scale_down(function, remove)

    def scale_up(self, function: str, add: Union[int, List[FunctionReplica]]):
        yield from self.faas.scale_up(function, add)
