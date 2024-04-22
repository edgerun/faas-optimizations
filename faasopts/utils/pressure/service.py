from typing import List

import pandas as pd

from faasopts.autoscalers.base.pressure.osmotic import ScaleScheduleEvent


class PressureService:

    def save(self, pressure_values: pd.DataFrame) -> str:
        raise NotImplementedError()

    def save_delete_results(self, delete_results: List[ScaleScheduleEvent]) -> str:
        raise NotImplementedError()


