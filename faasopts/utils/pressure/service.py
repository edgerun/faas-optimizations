import json
from typing import List

import pandas as pd
from faas.system import Metrics


class PressureService:

    def __init__(self, metrics: Metrics):
        self.metrics = metrics

    def publish_pressure_values(self, pressure_values: pd.DataFrame, zone: str):
        value = {'value': pressure_values.to_dict()}
        metric = f'pressure/{zone}'
        self.metrics.log(metric, value)

    def wait_for_all_pressures(self, zones: List[str]) -> pd.DataFrame:
        raise NotImplementedError()


class K8sPressureService(PressureService):

    def __init__(self, metrics: Metrics, rds):
        super().__init__(metrics)
        self.rds = rds

    def wait_for_all_pressures(self, zones: List[str]) -> pd.DataFrame:
        r = self.rds._rds
        p = r.pubsub(ignore_subscribe_messages=True)
        for zone in zones:
            p.subscribe(f'pressure/{zone}')
        results = []
        for item in p.listen():
            if item['type'] == 'message':
                result = item['data'].split(' ')[-1]
                result = json.loads(result)['value']
                result_df = pd.DataFrame(result)
                results.append(result_df)

                if len(results) == 3:
                    break  # All worker processes have finished, stop waiting
        return pd.concat(results)
