import logging
from collections import defaultdict
from typing import Optional, Callable, Dict

import pandas as pd
from faas.context import PlatformContext, FunctionReplicaFactory
from faas.system import FunctionReplica, Metrics
from faas.util.constant import zone_label

from faasopts.autoscalers.api import BaseAutoscaler
from faasopts.utils.pressure.api import PressureAutoscalerParameters
from faasopts.utils.pressure.calculation import PressureInput, PressureFunction
from faasopts.utils.pressure.service import PressureService

logger = logging.getLogger(__name__)


class PressureAutoscaler(BaseAutoscaler):

    def __init__(self, ctx: PlatformContext, parameters: PressureAutoscalerParameters,
                 gateway: FunctionReplica,
                 replica_factory: FunctionReplicaFactory, now: Callable[[], float], pressure_service: PressureService,
                 pressure_functions: Dict[str, PressureFunction], metrics: Metrics):
        self.ctx = ctx
        self.parameters = parameters
        self.gateway = gateway
        self.cluster = self.gateway.labels[zone_label]
        self.replica_factory = replica_factory
        self.now = now
        self.pressure_service = pressure_service
        self.pressure_functions = pressure_functions
        self.metrics = metrics

    def run(self) -> Optional[pd.DataFrame]:
        logger.info("start to figure scale up out")
        ctx = self.ctx
        pressure_values: pd.DataFrame = self.calculate_pressure_per_fn(ctx)
        if pressure_values is None:
            logger.info("No pressure values calculated, no further execution")
            return None
        # Store calculated pressure values such that global scheduler can retrieve it
        return pressure_values

    def calculate_pressure_per_fn(self, ctx: PlatformContext) -> Optional[pd.DataFrame]:
        """
        Results contains for each deployed function in the region the pressure value (grouped by client zones)
        Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
        """
        for function, fn_parameters in self.parameters.function_parameters.items():
            gateway = self.gateway
            now = self.now()
            past = now - fn_parameters.lookback
            traces = ctx.trace_service.get_traces_api_gateway(gateway.node.name, past, now, response_status=200)
            traces = traces[traces['function'] == function]
            gateway_node = ctx.node_service.find(gateway.node.name)
            zone = gateway_node.labels[zone_label]
            data = defaultdict(list)
            if len(traces) == 0:
                logger.info(f'Found no traces for gateway on node {gateway.node.name}')
                return None
            pressure_values = []

            internal_pressure_values = self.find_pressures_for_internal_clients(traces)
            if internal_pressure_values is not None and len(internal_pressure_values) > 0:
                pressure_values.append(internal_pressure_values)

            external_pressure_values = self.calculate_pressures_external_clients(traces)
            if external_pressure_values is not None and len(external_pressure_values) > 0:
                pressure_values.append(external_pressure_values)

            if (internal_pressure_values is None or len(internal_pressure_values) == 0) and (
                    external_pressure_values is None or len(external_pressure_values) == 0):
                logger.info(f'No pressure values calculated for zone {zone}')
                return None

            pressure_values = pd.concat(pressure_values)
            deployments = ctx.deployment_service.get_deployments()
            for deployment in deployments:
                fn_name = deployment.fn_name
                df = pressure_values[pressure_values['fn'] == fn_name]
                for client_zone in df['client_zone'].unique():
                    df_client_zone = df[df['client_zone'] == client_zone]
                    avg = df_client_zone['pressure'].mean()
                    median = df_client_zone['pressure'].median()
                    std = df_client_zone['pressure'].std()
                    amin = df_client_zone['pressure'].min()
                    amax = df_client_zone['pressure'].max()

                    data['fn'].append(fn_name)
                    data['fn_zone'].append(zone)
                    data['client_zone'].append(client_zone)
                    data['pressure_avg'].append(avg)
                    data['pressure_median'].append(median)
                    data['pressure_std'].append(std)
                    data['pressure_min'].append(amin)
                    data['pressure_max'].append(amax)

            gateway_pressure = pd.DataFrame(data=data)
            mean_pressure = gateway_pressure['pressure_avg'].mean()
            logger.info(f"Avg Pressure gateway ({gateway.node.name}) over all deployments: {mean_pressure}")
            return pressure_values.groupby(['fn', 'fn_zone', 'client_zone']).mean()

    def find_pressures_for_internal_clients(self, traces: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
        """
        cluster = self.cluster
        internally_spawned_traces = traces[traces['dest_zone'] == self.cluster]
        internally_spawned_traces = internally_spawned_traces[internally_spawned_traces['origin_zone'] == cluster]
        clients = internally_spawned_traces['client'].unique()

        dfs = []
        for client in clients:
            for_client = self.pressures_for_client(client, cluster, traces)
            if for_client is not None:
                dfs.append(for_client)
            break

        if len(dfs) == 0:
            return None
        df = pd.concat(dfs)
        df['client'] = f'gateway-{cluster}'
        return df

    def calculate_pressures_external_clients(self, traces: pd.DataFrame) -> Optional[pd.DataFrame]:
        ctx = self.ctx
        gateway = self.gateway

        pressure_values = []
        gateway_node = gateway.node
        gateway_zone = gateway_node.labels[zone_label]
        external_clients = traces[traces['origin_zone'] != gateway_zone][
            ['client', 'origin_zone']]
        seen_zones = set()
        unique_externals = external_clients.value_counts()
        # iterate over all external clients but only calculate the pressure once for each zone
        #
        for row in unique_externals.keys():
            client = row[0]
            origin_zone = row[1]
            if origin_zone not in seen_zones:
                pressures = self.pressures_for_client(client, origin_zone, traces)
                if pressures is None:
                    continue
                pressures['client'] = f'gateway-{gateway_zone}'
                pressure_values.append(pressures)
                seen_zones.add(origin_zone)

        if len(pressure_values) > 0:
            df = pd.concat(pressure_values)
            return df
        else:
            return None

    def pressures_for_client(self, client: str, client_zone: str, traces: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
        """
        ctx = self.ctx
        data = defaultdict(list)
        gateway_node = self.gateway.node
        zone = gateway_node.labels[zone_label]
        deployments = ctx.deployment_service.get_deployments()

        # client has format of <replica-id>:function:client_nr
        client_replica_id = client.split(':')[0]

        for deployment in deployments:
            fn_name = deployment.fn_name
            pods = ctx.replica_service.get_function_replicas_of_deployment(deployment.original_name)
            if len(pods) == 0:
                continue

            p = self.pressure_by_client_on_function(
                client,
                client_replica_id,
                fn_name,
                traces,
                ctx,
            )
            data['pressure'].append(p)
            data['client'].append(client)
            data['client_replica_id'].append(client_replica_id)
            data['client_zone'].append(client_zone)
            data['fn'].append(fn_name)
            data['fn_zone'].append(zone)

        if len(data) > 0:
            return pd.DataFrame(data=data)
        else:
            return None

    def pressure_by_client_on_function(
            self,
            client: str,
            client_replica_id: str,
            function: str,
            traces: pd.DataFrame,
            ctx: PlatformContext,
    ) -> float:
        """
        Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
        """

        now = self.now()
        pressure_input = PressureInput(
            parameters=self.parameters.function_parameters[function],
            client=client,
            client_replica_id=client_replica_id,
            gateway=self.gateway,
            function=function,
            now=now,
            traces=traces,
            ctx=ctx
        )
        val = 1
        for pressure in self.parameters[function].pressures:
            val *= self.pressure_functions[pressure].calculate_weighted_pressure(pressure_input)

        return val
