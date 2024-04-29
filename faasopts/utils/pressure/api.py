import logging
from dataclasses import dataclass
from typing import List, Optional, Dict

from dataclasses_json import dataclass_json
from faas.system import FunctionReplica

from faasopts.utils.pressure.calculation import LogisticFunctionParameters

logger = logging.getLogger(__name__)

@dataclass_json
@dataclass
class PressureFunctionParameters:
    # this thresholds are used to determine functions under pressure
    max_threshold: float
    min_threshold: float
    function_requirment: float
    # either 'latency' (in ms) or 'rtt' (in s)
    target_time_measure: str
    max_replicas: int
    pressure_names: List[str]
    logistic_function_parameters: Optional[LogisticFunctionParameters] = None
    percentile_duration: float = 90


@dataclass_json
@dataclass
class PressureAutoscalerParameters:
    function_parameters: Dict[str, PressureFunctionParameters]
    max_latency: float
    lookback: int


@dataclass_json
@dataclass
class ScaleScheduleEvent:
    ts: float
    fn: str
    replica: FunctionReplica
    origin_zone: str
    target_zone: str
    delete: bool = False
