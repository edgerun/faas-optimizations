import logging
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict

from dataclasses_json import dataclass_json
from faas.context import PlatformContext
from faas.system import FunctionReplica

from faasopts.utils.pressure.calculation import LogisticFunctionParameters

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class PressureScalerParameters:
    violates_threshold: Callable[[PlatformContext, FunctionReplica], bool]
    violates_min_threshold: Callable[[PlatformContext, FunctionReplica], bool]
    # this thresholds are used to determine functions under pressure
    max_threshold: float
    min_threshold: float
    function_requirements: Dict[str, float]
    max_latency: float
    lookback: int
    pressures: List[str]
    max_containers: int
    # either latency (in ms) or rtt (in s)
    target_time_measure: str
    reconcile_interval: float
    percentile_duration: float = 90
    logistic_function_parameters: Optional[LogisticFunctionParameters] = None


@dataclass_json
@dataclass
class ScaleScheduleEvent:
    ts: float
    fn: str
    replica: FunctionReplica
    origin_zone: str
    target_zone: str
    delete: bool = False
