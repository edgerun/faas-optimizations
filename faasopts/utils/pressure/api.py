import copy
import logging
from dataclasses import dataclass
from typing import List, Dict

from dataclasses_json import dataclass_json
from faas.system import FunctionReplica

logger = logging.getLogger(__name__)

@dataclass_json
@dataclass
class PressureFunctionParameters:
    # this thresholds are used to determine functions under pressure
    max_threshold: float
    min_threshold: float
    function_requirement: float
    # either 'latency' (in ms) or 'rtt' (in s)
    target_time_measure: str
    max_replicas: int
    pressure_weights: Dict[str, float]
    a: float  # max
    b: float  # bottom
    c: float  # growth
    d: float
    offset: float  # offset of midpoint (values > 0 increase the y-value of the midpoint)
    lookback: int
    percentile_duration: float = 90

    def copy(self):
        return PressureFunctionParameters(
            self.max_threshold,
            self.min_threshold,
            self.function_requirement,
            self.target_time_measure,
            self.max_replicas,
            copy.deepcopy(self.pressure_weights),
            self.a,
            self.b,
            self.c,
            self.d,
            self.offset,
            self.lookback,
            self.percentile_duration
        )

@dataclass_json
@dataclass
class PressureAutoscalerParameters:
    # parameters per function (i.e., key: function, value: parameters for scaling)
    function_parameters: Dict[str, PressureFunctionParameters]
    max_latency: float

    def copy(self):
        copied_function_parameters = self.function_parameters
        for fn, parameters in self.function_parameters.items():
            copied_function_parameters[fn] = parameters.copy()

        return PressureAutoscalerParameters(copied_function_parameters, self.max_latency)


@dataclass_json
@dataclass
class ScaleScheduleEvent:
    ts: float
    fn: str
    replica: FunctionReplica
    origin_zone: str
    target_zone: str
    delete: bool = False
