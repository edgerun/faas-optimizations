"""
Basically all of this code stems from the jacob-thesis branch from the faas-sim project.
Thanks, @jjnp for this implementation.
"""
import abc
import logging
import math
import statistics
from collections import defaultdict
from typing import Dict, List, Callable

import numpy as np
from faas.context import PlatformContext, FunctionReplicaService, TraceService
from faas.system import FunctionReplicaState, FunctionReplica, Metrics
from faas.system.loadbalancer import LocalizedLoadBalancer

logger = logging.getLogger(__name__)


class LeastResponseTimeMetricProvider:
    def __init__(self, context: PlatformContext, now: Callable[[], float], replica_ids: List[str],
                 window: float = 10.0):
        self.context = context
        self.now = now
        self.trace_service: TraceService = context.trace_service
        self.replica_service: FunctionReplicaService = context.replica_service
        self.window = window
        self.replica_ids = replica_ids
        self.rts = dict()
        self.last_record_timestamps = dict()
        self._init_values()

    def add_replica(self, replica_id: str):
        self.replica_ids.append(replica_id)
        self.last_record_timestamps[replica_id] = -1
        if len(self.rts.values()) > 0:
            self.rts[replica_id] = float(statistics.median(list(self.rts.values())))
        else:
            self.rts[replica_id] = 0.05

    def remove_replica(self, replica_id: str):
        self.replica_ids.remove(replica_id)
        del self.rts[replica_id]
        del self.last_record_timestamps[replica_id]

    def _init_values(self):
        for r in self.replica_ids:
            self.rts[r] = 0.05
            self.last_record_timestamps[r] = -1

    def record_response_time(self, replica_id: str, new_response_time: float, new_ts):
        if replica_id not in self.replica_ids:
            return
        last_record_timestamp = self.last_record_timestamps[replica_id]
        if last_record_timestamp == -1:
            self.last_record_timestamps[replica_id] = self.now()
            self.rts[replica_id] = new_response_time
            return
        time_delta = new_ts - last_record_timestamp
        alpha = 1.0 - math.exp(-time_delta / self.window)
        next_avg_rt = (alpha * new_response_time) + ((1.0 - alpha) * self.rts[replica_id])
        self.last_record_timestamps[replica_id] = new_ts
        self.rts[replica_id] = next_avg_rt

    def get_response_times(self) -> Dict[str, float]:
        for replica_id in self.replica_ids:
            replica = self.replica_service.get_function_replica_by_id(replica_id)
            traces = self.trace_service.get_traces_for_function_image(
                function=replica.fn_name,
                function_image=replica.container.image,
                response_status=200,
                start=self.now() - self.window,
                end=self.now()
            )
            if traces is None or len(traces) == 0:
                continue
            traces = traces.sort_values(by='ts', ascending=False)
            newest_trace = traces.iloc[0]
            self.record_response_time(replica_id, newest_trace['rtt'], newest_trace['ts'])
        return self.rts


class WeightCalculator(abc.ABC):

    def calculate_weights(self) -> Dict[str, Dict[str, float]]: ...

    def update(self, function: str, replica_ids: List[str]): ...


class LeastResponseTimeWeightCalculator(WeightCalculator):

    def __init__(self, context: PlatformContext, now: Callable[[], float], scaling: float,
                 lrt_window: float = 45,
                 max_weight=100):
        self.context = context
        self.now = now
        self.window = lrt_window
        self.scaling = scaling
        self.max_weight = max_weight
        self.lrt_providers: Dict[str, LeastResponseTimeMetricProvider] = {}
        self.weights: Dict[str, Dict[str, float]] = defaultdict(dict)

    def update(self, function: str, replica_ids: List[str]):
        self.lrt_providers[function] = LeastResponseTimeMetricProvider(self.context, self.now, replica_ids, self.window)

    def calculate_weights(self) -> Dict[str, Dict[str, float]]:
        fn_weights = {}
        for function in self.lrt_providers.keys():

            response_times = self.lrt_providers[function].get_response_times()
            if len(response_times) < 1:
                # fn_weights[function] = self.weights[function]
                continue
            else:
                min_weight = min(response_times.values())
                for r_id, rt in response_times.items():
                    w = int(round(max(1.0, pow(10 / (rt / min_weight), self.scaling))))
                    self.weights[function][r_id] = w
                fn_weights[function] = self.weights[function]

        return fn_weights

class SmoothLrtWeightCalculator(WeightCalculator):

    def __init__(self, context: PlatformContext, now: Callable[[], float], scaling: float,
                 lrt_window: float = 45,
                 max_weight=100):
        self.context = context
        self.max_weight = max_weight
        self.now = now
        self.window = lrt_window
        self.scaling = scaling
        self.max_weight = max_weight
        self.lrt_providers: Dict[str, LeastResponseTimeMetricProvider] = {}
        self.weights: Dict[str, Dict[str, float]] = defaultdict(dict)

    def update(self, function: str, replica_ids: List[str]):
        self.lrt_providers[function] = LeastResponseTimeMetricProvider(self.context, self.now, replica_ids, self.window)

    def calculate_weights(self) -> Dict[str, Dict[str, float]]:
        fn_weights = {}
        for function_name in self.lrt_providers.keys():
            weights = {}
            response_times = self.lrt_providers[function_name].get_response_times()
            if len(response_times) < 1:
                continue
            min_response_time = float(np.min(list(response_times.values())))

            for r_id, rt in response_times.items():
                weight = float(np.max([1, self.max_weight / math.pow((rt / min_response_time), self.scaling)]))
                weights[r_id] = weight
            fn_weights[function_name] = weights
        return fn_weights

class RoundRobinWeightCalculator(WeightCalculator):

    def __init__(self):
        self.functions = {}

    def calculate_weights(self) -> Dict[str, Dict[str, float]]:
        fn_weights = {}
        for function in self.functions.keys():
            weights = {}
            for replica_id in self.functions[function]:
                weights[replica_id] = 1
            fn_weights[function] = weights
        return fn_weights

    def update(self, function: str, replica_ids: List[str]):
        self.functions[function] = replica_ids




class WrrUpdater(LocalizedLoadBalancer):

    def __init__(self, context: PlatformContext, cluster: str, metrics: Metrics,
                 weight_calculator: WeightCalculator) -> None:
        super().__init__(context, cluster)
        self.metrics = metrics
        self.weight_calculator = weight_calculator

    def update(self):
        weights = self.calculate_weights()
        self.metrics.log('wrr-weights', 'update', **weights)
        self.set_weights(weights)

    def set_weights(self, weights: Dict[str, Dict[str, float]]):
        ...

    def get_running_replica_ids(self, replicas: List[FunctionReplica]):
        replica_ids = [r.replica_id for r in replicas if r.state == FunctionReplicaState.RUNNING]
        return replica_ids

    def _sync_replica_state(self):
        managed_functions = self.get_functions()
        for function in managed_functions:
            function_name = function.name
            replicas = self.get_running_replicas(function_name)
            replica_ids = self.get_running_replica_ids(replicas)
            self.weight_calculator.update(function_name, replica_ids)

    def calculate_weights(self) -> Dict[str, Dict[str, float]]:
        self._sync_replica_state()
        return self.weight_calculator.calculate_weights()
