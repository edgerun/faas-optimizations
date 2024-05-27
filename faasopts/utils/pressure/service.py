import time
from typing import List

import pandas as pd
import redis
from galileofaas.connections import RedisClient


class PressureService:

    def publish_pressure_values(self, pressure_values: pd.DataFrame, zone: str):
        raise NotImplementedError()

    def wait_for_all_pressures(self, zones: List[str]) -> pd.DataFrame:
        raise NotImplementedError()


class K8sPressureService(PressureService):

    def __init__(self, rds: RedisClient):
        self.rds = rds

    def publish_pressure_values(self, pressure_values: pd.DataFrame, zone: str):
        ts = time.time()
        channel = 'galileo/events'
        event = f'pressure/{zone}'
        body = pressure_values.to_json()
        msg = f'{ts} {event} {body}'
        self.rds.publish_async(channel, msg)

    def wait_for_all_pressures(self, zones: List[str]) -> pd.DataFrame:
        r: redis.Redis = self.rds._rds
        p = r.pubsub(ignore_subscribe_messages=True)
        for zone in zones:
            p.subscribe(f'pressure/{zone}')
        results = []
        for item in p.listen():
            if item['type'] == 'message':
                result = item['data'].split(' ')[-1]
                result_df = pd.read_json(result)
                results.append(result_df)

                if len(results) == 3:
                    break  # All worker processes have finished, stop waiting
        return pd.concat(results)
