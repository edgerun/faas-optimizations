import logging
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict

import pandas as pd
from dataclasses_json import dataclass_json
from faas.context import PlatformContext, FunctionReplicaFactory
from faas.system import FunctionReplica
from faas.util.constant import zone_label

from faasopts.autoscalers.api import BaseAutoscaler
from faasopts.utils.pressure.calculation import LogisticFunctionParameters
from faasopts.utils.pressure.service import OsmoticService

logger = logging.getLogger(__name__)

PRESSURE_TARGET_GATEWAY_ID = 'pressure-target-gateway-id'
DELETE_RESULTS_ID = 'delete-results-id'
PRESSURE_ID = 'pressure-id'
PRESSURE_ORIGIN_GATEWAY_ID = 'pressure-origin-gateway-id'

@dataclass
class OsmoticScalerParameters:
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
    percentile_duration: float = 90
    deployment_pattern: str = '-deployment'
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


def get_average_requests_over_replicas(fn: str, traces: pd.DataFrame):
    """
    Calculates the average number of requests over all replicas of one function.
    :param fn: function name
    :return:
    """
    df = traces[traces['function'] == fn]
    return df.groupby('container_id').count()['ts'].mean().mean()


class OsmoticAutoscaler(BaseAutoscaler):

    def __init__(self, ctx: PlatformContext, parameters: OsmoticScalerParameters, gateway: FunctionReplica,
                 replica_factory: FunctionReplicaFactory, now: Callable[[], float], pressure_service: OsmoticService):
        self.ctx = ctx
        self.parameters = parameters
        self.gateway = gateway
        self.cluster = self.gateway.labels[zone_label]
        self.replica_factory = replica_factory
        self.now = now
        self.pressure_service = pressure_service

    def run(self) -> Optional[pd.DataFrame]:
        logger.info("start to figure scale up out")
        ctx = self.ctx
        pressure_values: pd.DataFrame = self.calculate_pressure_per_fn_per_zone(ctx)
        if pressure_values is None:
            logger.info("No pressure values calculated, no further execution")
            return None
        # Store calculated pressure values such that global scheduler can retrieve it
        return pressure_values
