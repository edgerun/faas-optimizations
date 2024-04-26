from typing import List

import pandas as pd

from faasopts.autoscalers.base.pressure.autoscaler import ScaleScheduleEvent


class OsmoticService:

    def save_pressure_values(self, pressure_values: pd.DataFrame) -> str:
        raise NotImplementedError()

    def save_delete_results(self, delete_results: List[ScaleScheduleEvent]) -> str:
        raise NotImplementedError()

    def get_pressure_values(self, pressure_id: str) -> pd.DataFrame:
        raise NotImplementedError()

    def get_delete_results(self, delete_results_id: str) -> List[ScaleScheduleEvent]:
        raise NotImplementedError()

    def wait_for_all_pressures(self) -> pd.DataFrame:
        raise NotImplementedError()
