"""
Basically all of this code stems from the jacob-thesis branch from the faas-sim project.
Thanks, @jjnp for this implementation.
"""
import abc
import logging
import math
import statistics
from collections import defaultdict
from typing import Dict, List, Callable, Tuple

import numpy as np
from faas.context import PlatformContext, FunctionReplicaService, TraceService
from faas.system import FunctionReplicaState, FunctionReplica, Metrics
from faas.system.loadbalancer import LocalizedLoadBalancerOptimizer, GlobalLoadBalancerOptimizer
from faas.util.constant import pod_type_label, function_type_label, api_gateway_type_label, zone_label
from faas.util.rwlock import ReadWriteLock

logger = logging.getLogger(__name__)


class LeastResponseTimeMetricProvider:
    def __init__(self, function: str, context: PlatformContext, now: Callable[[], float], replica_ids: List[str],
                 window: float = 10.0):
        self.function = function
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
        try:
            self.replica_ids.remove(replica_id)
            del self.rts[replica_id]
            del self.last_record_timestamps[replica_id]
        except (KeyError, ValueError) as e:
            logger.warning(f'Wanted to delete {replica_id} but was not present. {e}')

    def _init_values(self):
        for r in self.replica_ids:
            self.rts[r] = 0.05
            self.last_record_timestamps[r] = -1

    def record_response_time(self, replica_id: str, new_response_time: float, new_ts):
        if replica_id not in self.replica_ids or replica_id not in self.last_record_timestamps:
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
            now = self.now()
            if replica.labels[pod_type_label] == function_type_label:
                traces = self.trace_service.get_traces_for_function_image(
                    function=replica.fn_name,
                    function_image=replica.container.image,
                    response_status=200,
                    start=now - self.window,
                    end=now
                )
                if traces is None or len(traces) == 0:
                    continue
                traces = traces.sort_values(by='ts', ascending=False)
                traces = traces[traces['replica_id'] == replica_id]
                if len(traces) == 0:
                    continue
                else:
                    newest_trace = traces.iloc[0]
                    self.record_response_time(replica_id, newest_trace['rtt'], newest_trace['ts'])
            if replica.labels[pod_type_label] == api_gateway_type_label:
                try:
                    traces = self.trace_service.get_traces_api_gateway(replica.node.name, now - self.window,
                                                                       now, 200)
                    traces = traces[traces['function'] == self.function]
                    if len(traces) == 0:
                        self.record_response_time(replica_id, 1, now - 1)
                    else:
                        # in case it's an API gateway, we want to aggregate over all instances behind it
                        for idx, row in traces.iterrows():
                            self.record_response_time(replica_id, row['rtt'], row['ts'])
                except Exception as e:
                    logger.error(e)
        return self.rts

    def contains(self, replica_id: str):
        try:
            self.replica_ids.index(replica_id)
            return True
        except ValueError:
            return False


class WeightCalculator(abc.ABC):

    def calculate_weights(self) -> Dict[str, Dict[str, float]]: ...

    def update(self, function: str, replica_ids: List[str]): ...

    def add_replica(self, function: str, replica: FunctionReplica): ...

    def remove_replica(self, function: str, replica: FunctionReplica): ...

    def add_replicas(self, function: str, replicas: List[FunctionReplica]): ...

    def remove_replicas(self, function: str, replicas: List[FunctionReplica]): ...

    def get_replicas_ids(self, function: str) -> List[str]: ...

    def manages_functions(self, function: str) -> bool:  ...


class LeastResponseTimeWeightCalculator(WeightCalculator):

    def __init__(self, context: PlatformContext, now: Callable[[], float], scaling: float,
                 lrt_window: float = 45,
                 max_weight=100):
        self.context = context
        self.now = now
        self.window = lrt_window
        self.scaling = scaling
        self.max_weight = max_weight
        self.lock = ReadWriteLock()
        self.lrt_providers: Dict[str, LeastResponseTimeMetricProvider] = {}
        self.weights: Dict[str, Dict[str, float]] = defaultdict(dict)

    def update(self, function: str, replica_ids: List[str]):
        self.lrt_providers[function] = LeastResponseTimeMetricProvider(function, self.context, self.now, replica_ids,
                                                                       self.window)

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

    def add_replica(self, function: str, replica: FunctionReplica):
        with self.lock.lock.gen_wlock():
            if self.lrt_providers.get(function) is None:
                self.lrt_providers[function] = LeastResponseTimeMetricProvider(function, self.context, self.now,
                                                                               [replica.replica_id],
                                                                               self.window)
            else:
                self.lrt_providers[function].add_replica(replica.replica_id)

    def remove_replica(self, function: str, replica: FunctionReplica):
        with self.lock.lock.gen_wlock():
            if self.lrt_providers.get(function) is None:
                self.lrt_providers[function] = LeastResponseTimeMetricProvider(function, self.context, self.now,
                                                                               [], self.window)
            else:
                self.lrt_providers[function].remove_replica(replica.replica_id)

    def add_replicas(self, function: str, replicas: List[FunctionReplica]):
        for replica in replicas:
            self.add_replica(function, replica)

    def remove_replicas(self, function: str, replicas: List[FunctionReplica]):
        for replica in replicas:
            self.remove_replica(function, replica)

    def get_replicas_ids(self, function: str) -> List[str]:
        return list(self.weights[function].keys())

    def manages_functions(self, function: str) -> bool:
        return self.weights.get(function) is not None


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
        self.lock = ReadWriteLock()
        self.weights: Dict[str, Dict[str, float]] = defaultdict(dict)

    def update(self, function: str, replica_ids: List[str]):
        self.lrt_providers[function] = LeastResponseTimeMetricProvider(function, self.context, self.now, replica_ids,
                                                                       self.window)

    def calculate_weights(self) -> Dict[str, Dict[str, float]]:
        fn_weights = {}
        for function_name in self.lrt_providers.keys():
            weights = {}
            response_times = self.lrt_providers[function_name].get_response_times()
            if len(response_times) < 1:
                # continue
                min_response_time = 1
                replica_ids = self.get_replicas_ids(function_name)

                for replica_id in replica_ids:
                    weights[replica_id] = 1
            else:
                min_response_time = float(np.min(list(response_times.values())))

                for r_id, rt in response_times.items():
                    weight = float(np.max([1, self.max_weight / math.pow((rt / min_response_time), self.scaling)]))
                    weights[r_id] = weight

            fn_weights[function_name] = weights
        self.weights = fn_weights
        return fn_weights

    def add_replica(self, function: str, replica: FunctionReplica):
        with self.lock.lock.gen_wlock():
            if self.lrt_providers.get(function) is None:
                self.lrt_providers[function] = LeastResponseTimeMetricProvider(function, self.context, self.now,
                                                                               [replica.replica_id],
                                                                               self.window)
            else:
                self.lrt_providers[function].add_replica(replica.replica_id)

    def remove_replica(self, function: str, replica: FunctionReplica):
        with self.lock.lock.gen_wlock():
            if self.lrt_providers.get(function) is None:
                self.lrt_providers[function] = LeastResponseTimeMetricProvider(function, self.context, self.now,
                                                                               [], self.window)
            else:
                self.lrt_providers[function].remove_replica(replica.replica_id)

    def add_replicas(self, function: str, replicas: List[FunctionReplica]):
        for replica in replicas:
            self.add_replica(function, replica)

    def remove_replicas(self, function: str, replicas: List[FunctionReplica]):
        for replica in replicas:
            self.remove_replica(function, replica)

    def get_replicas_ids(self, function: str) -> List[str]:
        return list(self.weights[function].keys())

    def manages_functions(self, function: str) -> bool:
        return self.weights.get(function) is not None


class RoundRobinWeightCalculator(WeightCalculator):

    def __init__(self):
        self.functions: Dict[str, List[str]] = defaultdict(list)
        self.weights: Dict[str, List[str]] = defaultdict(list)

    def calculate_weights(self) -> Dict[str, Dict[str, float]]:
        fn_weights = {}
        for function in self.functions.keys():
            weights = {}
            for replica_id in self.functions[function]:
                weights[replica_id] = 1
            fn_weights[function] = weights
        self.weights = fn_weights
        return fn_weights

    def update(self, function: str, replica_ids: List[str]):
        self.functions[function] = replica_ids

    def add_replica(self, function: str, replica: FunctionReplica):
        self.functions[function].append(replica.replica_id)

    def remove_replica(self, function: str, replica: FunctionReplica):
        self.functions[function].remove(replica.replica_id)

    def add_replicas(self, function: str, replicas: List[FunctionReplica]):
        for replica in replicas:
            self.add_replica(function, replica)

    def remove_replicas(self, function: str, replicas: List[FunctionReplica]):
        for replica in replicas:
            self.remove_replica(function, replica)

    def get_replicas_ids(self, function: str) -> List[str]:
        return list(self.weights[function])

    def manages_functions(self, function: str) -> bool:
        return self.weights.get(function) is not None


class WrrOptimizer(LocalizedLoadBalancerOptimizer):

    def __init__(self, context: PlatformContext, cluster: str, metrics: Metrics,
                 weight_calculator: WeightCalculator) -> None:
        super().__init__(context, cluster)
        self.metrics = metrics
        self.weight_calculator = weight_calculator
        # context.replica_service.register(self.observer)

    def update(self):
        weights = self.calculate_weights()
        self.metrics.log('wrr-weights', self.cluster, **weights)
        self.set_weights(weights)

    def _get_function(self, replica: FunctionReplica) -> List[Tuple[str, FunctionReplica]]:
        if replica.state is not FunctionReplicaState.RUNNING:
            return []
        if replica.labels.get(pod_type_label) is None:
            return []
        if replica.labels[pod_type_label] == function_type_label:
            if replica.node.cluster != self.cluster:
                # function replica but not running in same cluster, we have to add load balancer of the other cluster
                node_labels = {
                    zone_label: replica.node.cluster
                }
                labels = {
                    pod_type_label: api_gateway_type_label
                }
                replicas = self.context.replica_service.find_function_replicas_with_labels(labels=labels,
                                                                                           node_labels=node_labels,
                                                                                           state=FunctionReplicaState.RUNNING)
                # we assume that only one load balancer runs per cluster
                if len(replicas) == 0:
                    return []
                lb_replica = replicas[0]
                return [(replica.function.name, lb_replica)]
            else:
                # function replica in same cluster
                return [(replica.function.name, replica)]
        if replica.labels[pod_type_label] == api_gateway_type_label:
            if replica.node.cluster == self.cluster:
                # we ignore replicas that are load balancers running in the same cluster (-> it would just add itself)
                return []
            # check if in the load balancer's zone there exists function replicas
            # and return all functions that have an instance running
            node_labels = {
                zone_label: replica.node.cluster
            }
            labels = {
                pod_type_label: function_type_label
            }
            functions = set()
            replicas = self.context.replica_service.find_function_replicas_with_labels(labels=labels,
                                                                                       node_labels=node_labels,
                                                                                       state=FunctionReplicaState.RUNNING)
            for replica in replicas:
                functions.add(replica.fn_name)

            return [(f, replica) for f in functions]

    def add_replica(self, replica: FunctionReplica):
        functions = self._get_function(replica)
        for function, actual_replica in functions:
            self.weight_calculator.add_replica(function, actual_replica)

    def remove_replica(self, replica: FunctionReplica):
        functions = self._get_function(replica)
        for function, actual_replica in functions:
            self.weight_calculator.remove_replica(function, actual_replica)

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
            if not self.weight_calculator.manages_functions(function_name):
                replica_ids = self.get_running_replica_ids(replicas)
                self.weight_calculator.update(function_name, replica_ids)
            else:
                try:
                    current_replica_ids = self.get_running_replica_ids(replicas)
                    weight_calculator_replica_ids = self.weight_calculator.get_replicas_ids(function_name)
                    if not set(current_replica_ids) == set(weight_calculator_replica_ids):
                        for replica in replicas:
                            # A new replica as added
                            if replica.replica_id not in weight_calculator_replica_ids:
                                self.weight_calculator.add_replica(function_name, replica)
                        for r_id in weight_calculator_replica_ids:
                            # A replica was removed
                            if r_id not in current_replica_ids:
                                replica = self.context.replica_service.get_function_replica_by_id(r_id)
                                self.weight_calculator.remove_replica(function_name, replica)
                except Exception as e:
                    logger.error(f'something went wrong syncing {e}')

    def calculate_weights(self) -> Dict[str, Dict[str, float]]:
        self._sync_replica_state()
        return self.weight_calculator.calculate_weights()

    def add_replicas(self, replicas: List[FunctionReplica]):
        for replica in replicas:
            self.add_replica(replica)

    def remove_replicas(self, replicas: List[FunctionReplica]):
        for replica in replicas:
            self.remove_replica(replica)


class GlobalWrrOptimizer(GlobalLoadBalancerOptimizer):

    def __init__(self, context: PlatformContext, metrics: Metrics,
                 weight_calculator: WeightCalculator) -> None:
        super().__init__(context)
        self.metrics = metrics
        self.weight_calculator = weight_calculator
        # context.replica_service.register(self.observer)

    def update(self):
        weights = self.calculate_weights()
        self.metrics.log('wrr-weights', 'global', **weights)
        self.set_weights(weights)

    def _get_function(self, replica: FunctionReplica) -> List[Tuple[str, FunctionReplica]]:
        if replica.state is not FunctionReplicaState.RUNNING:
            return []
        if replica.labels.get(pod_type_label) is None:
            return []
        if replica.labels[pod_type_label] == function_type_label:
            return [(replica.function.name, replica)]

    def add_replica(self, replica: FunctionReplica):
        functions = self._get_function(replica)
        for function, actual_replica in functions:
            self.weight_calculator.add_replica(function, actual_replica)

    def remove_replica(self, replica: FunctionReplica):
        functions = self._get_function(replica)
        for function, actual_replica in functions:
            self.weight_calculator.remove_replica(function, actual_replica)

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
            if not self.weight_calculator.manages_functions(function_name):
                replica_ids = self.get_running_replica_ids(replicas)
                self.weight_calculator.update(function_name, replica_ids)
            else:
                try:
                    current_replica_ids = self.get_running_replica_ids(replicas)
                    weight_calculator_replica_ids = self.weight_calculator.get_replicas_ids(function_name)
                    if not set(current_replica_ids) == set(weight_calculator_replica_ids):
                        for replica in replicas:
                            # A new replica as added
                            if replica.replica_id not in weight_calculator_replica_ids:
                                self.weight_calculator.add_replica(function_name, replica)
                        for r_id in weight_calculator_replica_ids:
                            # A replica was removed
                            if r_id not in current_replica_ids:
                                replica = self.context.replica_service.get_function_replica_by_id(r_id)
                                self.weight_calculator.remove_replica(function_name, replica)
                except Exception as e:
                    logger.error('something went wrong syncing', e)

    def calculate_weights(self) -> Dict[str, Dict[str, float]]:
        self._sync_replica_state()
        return self.weight_calculator.calculate_weights()

    def add_replicas(self, replicas: List[FunctionReplica]):
        for replica in replicas:
            self.add_replica(replica)

    def remove_replicas(self, replicas: List[FunctionReplica]):
        for replica in replicas:
            self.remove_replica(replica)
