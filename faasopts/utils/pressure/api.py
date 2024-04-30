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
    pressure_names: List[str]
    a: float  # max
    b: float  # bottom
    c: float  # growth
    d: float
    offset: float  # offset of midpoint (values > 0 increase the y-value of the midpoint)
    lookback: int
    percentile_duration: float = 90


@dataclass_json
@dataclass
class PressureAutoscalerParameters:
    # parameters per function (i.e., key: function, value: parameters for scaling)
    function_parameters: Dict[str, PressureFunctionParameters]
    max_latency: float


@dataclass_json
@dataclass
class ScaleScheduleEvent:
    ts: float
    fn: str
    replica: FunctionReplica
    origin_zone: str
    target_zone: str
    delete: bool = False
