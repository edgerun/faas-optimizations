import logging
from collections import defaultdict
from typing import Optional, Callable, List

import pandas as pd
from faas.context import PlatformContext, FunctionReplicaFactory
from faas.system import FunctionReplica, Metrics
from faas.util.constant import zone_label, function_label, pod_type_label, api_gateway_type_label

from faasopts.autoscalers.api import BaseAutoscaler
from faasopts.utils.infrastructure.filter import get_filtered_nodes_in_zone
from faasopts.utils.pressure.api import PressureAutoscalerParameters, PressureScaleScheduleEvent
from faasopts.utils.pressure.calculation import PressureInput, identify_above_max_pressure_deployments, \
    is_below_min_threshold, prepare_pressure_scale_schedule_events, create_pressure_functions

logger = logging.getLogger(__name__)


class PressureAutoscaler(BaseAutoscaler):

    def __init__(self, ctx: PlatformContext, parameters: PressureAutoscalerParameters,
                 zone: str,
                 replica_factory: FunctionReplicaFactory, now: Callable[[], float],
                 metrics: Metrics):
        self.ctx = ctx
        self.parameters = parameters
        self.zone = zone
        self.replica_factory = replica_factory
        self.now = now
        self.pressure_functions = create_pressure_functions(parameters)
        self.metrics = metrics
        self.local_scheduler_name = parameters.local_scheduler_name

    def run(self) -> Optional[pd.DataFrame]:
        ctx = self.ctx
        pressure_values: pd.DataFrame = self.calculate_pressure_per_fn(ctx)
        if pressure_values is None:
            logger.info("No pressure values calculated, no further execution")
            return None
        # Store calculated pressure values such that global scheduler can retrieve it
        return pressure_values

    def find_local_scale_actions(self, pressure_values: pd.DataFrame) -> List[PressureScaleScheduleEvent]:
        """
        In contrast
        :param pressure_values:
        :return:
        """
        zone_parameters = {self.zone: self.parameters}

        # contains
        below_min_pressure = is_below_min_threshold(pressure_values, self.ctx, zone_parameters)

        teardowns = defaultdict(int)
        for result in below_min_pressure:
            teardowns[result.deployment.name] += 1

        above_max_pressures = identify_above_max_pressure_deployments(pressure_values, teardowns, self.ctx,
                                                                      zone_parameters)

        local_scale_ups_per_fn = defaultdict(list)
        for result in above_max_pressures:
            if result.target_gateway.replica_id == result.origin_gateway.replica_id and result.origin_gateway.node.cluster == self.zone:
                # we only want to resolve pressure results that are above if the violation originates from the zone
                # that the autoscaler observes
                local_scale_ups_per_fn[result.deployment.name].append(result)

        local_teardowns_per_fn = defaultdict(list)
        for result in below_min_pressure:
            if result.target_gateway.replica_id == result.origin_gateway.replica_id and result.origin_gateway.node.cluster == self.zone:
                # we only want to resolve pressure results that are above if the violation originates from the zone
                # that the autoscaler observes
                local_teardowns_per_fn[result.deployment.name].append(result)
        actions = []
        for fn, results in local_teardowns_per_fn.items():
            for result in results:
                scale_down_rate = 1
                teardown_replicas = self.select_teardown_replicas(fn, scale_down_rate)
                actions.append(PressureScaleScheduleEvent(
                    ts=self.now(),
                    fn=fn,
                    replicas=teardown_replicas, delete=result.scale_down,
                    target_zone=result.target_gateway.labels[zone_label],
                    origin_zone=result.origin_gateway.labels[zone_label]))

        for fn, results in local_scale_ups_per_fn.items():
            # scale_up_rate = self.parameters.function_parameters[fn].scale_up_rate
            for result in results:
                scale_up_rate = 1
                event = prepare_pressure_scale_schedule_events(result.deployment, self.zone, self.zone,
                                                               self.local_scheduler_name, self.replica_factory,
                                                               self.now, scale_up_rate)
                ctr = 0
                for replica in event.replicas:
                    if self.replica_fits(replica, self.zone):
                        ctr += 1

                if ctr == 0:
                    logger.info("No space for replicas, let global scheduler handle this case")
                else:
                    replicas_to_scale = event.replicas[:ctr]
                    event.replicas = replicas_to_scale
                actions.append(event)
        return actions

    def calculate_pressure_per_fn(self, ctx: PlatformContext) -> Optional[pd.DataFrame]:
        """
        Results contains for each deployed function in the region the pressure value (grouped by client zones)
        Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
        """
        gateway = \
            ctx.replica_service.find_function_replicas_with_labels(labels={pod_type_label: api_gateway_type_label},
                                                                   node_labels={zone_label: self.zone})[0]
        pressure_values = []

        for function, fn_parameters in self.parameters.function_parameters.items():
            now = self.now()
            past = now - fn_parameters.lookback
            traces = ctx.trace_service.get_traces_api_gateway(gateway.node.name, past, now, response_status=200)
            traces = traces[~traces['client'].str.contains('load')]
            traces = traces[traces['function'] == function]
            gateway_node = ctx.node_service.find(gateway.node.name)
            zone = gateway_node.labels[zone_label]
            if len(traces) == 0:
                logger.info(f'Found no traces for gateway on node {gateway.node.name}')
                return None

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

        return pd.concat(pressure_values).groupby(['fn', 'fn_zone', 'client_zone']).mean()

    def find_pressures_for_internal_clients(self, traces: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
        """
        zone = self.zone
        internally_spawned_traces = traces[traces['dest_zone'] == self.zone]
        internally_spawned_traces = internally_spawned_traces[internally_spawned_traces['origin_zone'] == zone]
        clients = internally_spawned_traces['client'].unique()

        dfs = []
        for client in clients:
            for_client = self.pressures_for_client(client, zone, traces)
            if for_client is not None:
                dfs.append(for_client)
            break

        if len(dfs) == 0:
            return None
        df = pd.concat(dfs)
        df['client'] = f'gateway-{zone}'
        return df

    def calculate_pressures_external_clients(self, traces: pd.DataFrame) -> Optional[pd.DataFrame]:
        ctx = self.ctx
        gateway = \
            ctx.replica_service.find_function_replicas_with_labels(labels={pod_type_label: api_gateway_type_label},
                                                                   node_labels={zone_label: self.zone})[0]
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
        gateway = \
            ctx.replica_service.find_function_replicas_with_labels(labels={pod_type_label: api_gateway_type_label},
                                                                   node_labels={zone_label: self.zone})[0]
        gateway_node = gateway.node
        zone = gateway_node.labels[zone_label]
        deployments = ctx.deployment_service.get_deployments()

        # client has format of <replica-id>:function:client_nr
        client_replica_id = client.split(':')[0]

        for fn in self.parameters.function_parameters.keys():
            pods = ctx.replica_service.get_function_replicas_of_deployment(fn)
            if len(pods) == 0:
                continue

            p = self.pressure_by_client_on_function(
                client,
                client_replica_id,
                fn,
                traces,
                ctx,
            )
            data['pressure'].append(p)
            data['client'].append(client)
            data['client_replica_id'].append(client_replica_id)
            data['client_zone'].append(client_zone)
            data['fn'].append(fn)
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
        gateway = \
            ctx.replica_service.find_function_replicas_with_labels(labels={pod_type_label: api_gateway_type_label},
                                                                   node_labels={zone_label: self.zone})[0]
        now = self.now()
        pressure_input = PressureInput(
            parameters=self.parameters,
            client=client,
            client_replica_id=client_replica_id,
            gateway=gateway,
            function=function,
            now=now,
            traces=traces,
            ctx=ctx
        )
        val = 1
        for name, pressure in self.pressure_functions[function].items():
            weight = self.parameters.function_parameters[function].pressure_weights[name]
            pressure_value = pressure.calculate_pressure(pressure_input)
            val *= (pressure_value * weight)

        return val

    def select_teardown_replicas(self, fn: str, no_to_teardown: int):
        replicas = self.ctx.replica_service.find_function_replicas_with_labels(
            {function_label: fn}, node_labels={zone_label: self.zone})
        return replicas[:no_to_teardown]

    def replica_fits(self, replica: FunctionReplica, zone: str):
        available_nodes = get_filtered_nodes_in_zone(self.ctx, replica, zone)
        return len(available_nodes) > 0
