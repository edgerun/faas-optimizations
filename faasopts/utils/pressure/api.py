import copy
import logging
from dataclasses import dataclass
from typing import Dict, List

from dataclasses_json import dataclass_json
from faas.system import FunctionReplica

logger = logging.getLogger(__name__)

@dataclass_json
@dataclass
class PressureFunctionParameters:
    # this thresholds are used to determine functions under pressure
    max_threshold: float
    min_threshold: float
    # determines the optimal value (i.e., upper limit) of how long requests should take
    function_requirement: float
    # either 'latency' (in ms) or 'rtt' (in s)
    target_time_measure: str
    # the pressures here specified, are also used to calculate the pressure
    # i.e., only specify pressures that should be used
    pressure_weights: Dict[str, float]
    a: float  # max
    b: float  # bottom
    c: float  # growth
    d: float
    offset: float  # offset of midpoint (values > 0 increase the y-value of the midpoint)
    lookback: int  # monitoring to consider in the past <lookback> seconds
    percentile_duration: float = 90  # used for RTT pressure as percentile for aggregation
    scale_up_rate: int = 1

    def copy(self):
        return PressureFunctionParameters(
            self.max_threshold,
            self.min_threshold,
            self.function_requirement,
            self.target_time_measure,
            copy.deepcopy(self.pressure_weights),
            self.a,
            self.b,
            self.c,
            self.d,
            self.offset,
            self.lookback,
            self.percentile_duration,
            self.scale_up_rate
        )

@dataclass_json
@dataclass
class PressureAutoscalerParameters:
    # parameters per function (i.e., key: function, value: parameters for scaling)
    function_parameters: Dict[str, PressureFunctionParameters]
    # used when handling local actions
    local_scheduler_name: str

    def copy(self):
        copied_function_parameters = self.function_parameters
        for fn, parameters in self.function_parameters.items():
            copied_function_parameters[fn] = parameters.copy()

        return PressureAutoscalerParameters(copied_function_parameters, self.local_scheduler_name)


@dataclass_json
@dataclass
class PressureScaleScheduleEvent:
    ts: float
    fn: str
    replicas: List[FunctionReplica]
    origin_zone: str
    target_zone: str
    delete: bool = False
