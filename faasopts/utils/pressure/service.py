from typing import List

import pandas as pd

from faasopts.utils.pressure.api import PressureScaleScheduleEvent


class PressureService:

    def save_pressure_values(self, pressure_values: pd.DataFrame) -> str:
        raise NotImplementedError()

    def save_delete_results(self, delete_results: List[PressureScaleScheduleEvent]) -> str:
        raise NotImplementedError()

    def get_pressure_values(self, pressure_id: str) -> pd.DataFrame:
        raise NotImplementedError()

    def get_delete_results(self, delete_results_id: str) -> List[PressureScaleScheduleEvent]:
        raise NotImplementedError()

    def wait_for_all_pressures(self) -> pd.DataFrame:
        raise NotImplementedError()
