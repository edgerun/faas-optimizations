import abc
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable

import numpy as np
import pandas as pd
from faas.context import PlatformContext
from faas.system import FunctionReplica, FunctionDeployment
from faas.util.constant import pod_type_label, api_gateway_type_label, zone_label, function_label, pod_pending

from faasopts.utils.pressure.api import PressureAutoscalerParameters, PressureFunctionParameters, ScaleScheduleEvent

logger = logging.getLogger(__name__)


@dataclass
class LogisticFunctionParameters:
    a: float  # max
    b: float  # bottom
    c: float  # growth
    d: float
    offset: float  # offset of midpoint (values > 0 increase the y-value of the midpoint)


@dataclass
class PressureResult:
    target_gateway: FunctionReplica
    origin_gateway: FunctionReplica
    deployment: FunctionDeployment

def logistic_curve(x, a, b, c, d):
    """
    Logistic function with parameters a, b, c, d
    a is the curve's maximum value (top asymptote)
    b is the curve's minimum value (bottom asymptote)
    c is the logistic growth rate or steepness of the curve
    d is the x value of the sigmoid's midpoint
    """
    return ((a - b) / (1 + np.exp(-c * (x - d)))) + b


@dataclass
class PressureInput:
    parameters: PressureAutoscalerParameters
    client: str
    client_replica_id: str
    gateway: FunctionReplica
    function: str
    now: float
    traces: pd.DataFrame
    ctx: PlatformContext


class PressureFunction(abc.ABC):

    def calculate_pressure(self, pressure_input: PressureInput) -> float:
        raise NotImplementedError()

    def name(self) -> str:
        raise NotImplementedError()


def calculate_average_internal_distance(gateway: FunctionReplica, ctx: PlatformContext) -> float:
    gateway_node = gateway.node
    zone = gateway_node.labels[zone_label]
    nodes = ctx.node_service.find_nodes_in_zone(zone)
    distances = list(
        map(lambda c: ctx.network_service.get_latency(gateway_node.name, c.name), nodes)
    )
    return np.mean(distances)


class PressureRTTLogFunction(PressureFunction):

    def calculate_pressure(self, pressure_input: PressureInput) -> float:
        parameters = pressure_input.parameters
        client_replica_id = pressure_input.client_replica_id
        client = pressure_input.client
        gateway = pressure_input.gateway
        fn = pressure_input.function
        now = pressure_input.now
        ctx = pressure_input.ctx
        return pressure_rtt_log(parameters[fn], client, client_replica_id, gateway, fn, now, ctx)

    def name(self) -> str:
        return 'rtt_log'


class PressureRequestFunction(PressureFunction):

    def calculate_pressure(self, pressure_input: PressureInput) -> float:
        client = pressure_input.client
        fn = pressure_input.function
        traces = pressure_input.traces
        return pressure_by_num_requests_for_gateway(client, fn, traces)

    def name(self) -> str:
        return 'request'


class PressureNetworkDistanceFunction(PressureFunction):

    def calculate_pressure(self, pressure_input: PressureInput) -> float:
        client_replica_id = pressure_input.client_replica_id
        gateway = pressure_input.gateway
        ctx = pressure_input.ctx
        max_latency = ctx.network_service.get_max_latency()
        return pressure_by_network_distance(client_replica_id, max_latency, ctx, gateway)

    def name(self) -> str:
        return 'network_distance'


class PressureLatencyFulfillmentFunction(PressureFunction):

    def calculate_pressure(self, pressure_input: PressureInput) -> float:
        parameters = pressure_input.parameters
        client_replica_id = pressure_input.client_replica_id
        gateway = pressure_input.gateway
        fn = pressure_input.function
        ctx = pressure_input.ctx
        required_latency = parameters.function_requirements[fn]
        return pressure_latency_fulfillment(client_replica_id, required_latency, ctx, gateway)

    def name(self) -> str:
        return 'latency_fulfillment'


class PressureLatencyReqLogFunction(PressureFunction):

    def calculate_pressure(self, pressure_input: PressureInput) -> float:
        parameters = pressure_input.parameters
        client_replica_id = pressure_input.client_replica_id
        gateway = pressure_input.gateway
        fn = pressure_input.function
        ctx = pressure_input.ctx
        required_latency = parameters[fn].function_requirements[fn]
        return pressure_latency_req_log(parameters[fn], client_replica_id, required_latency, gateway, ctx)

    def name(self) -> str:
        return 'latency_req_log'


class PressureCpuUsageFunction(PressureFunction):

    def calculate_pressure(self, pressure_input: PressureInput) -> float:
        parameters = pressure_input.parameters
        now = pressure_input.now
        gateway = pressure_input.gateway
        fn = pressure_input.function
        ctx = pressure_input.ctx
        return pressure_cpu_usage(parameters.function_parameters[fn], fn, gateway, now, ctx)

    def name(self) -> str:
        return 'cpu_usage'


def pressure_rtt_log(parameters: PressureFunctionParameters, client: str, client_replica_id: str,
                     gateway: FunctionReplica, fn: str,
                     now: float, ctx: PlatformContext):
    a = parameters.a
    b = parameters.b
    c = parameters.c
    d = parameters.d
    offset = parameters.offset
    client_node_name = ctx.replica_service.get_function_replica_by_id(client_replica_id).node.name
    client_node = ctx.node_service.find(client_node_name)

    client_gateway_cluster = client_node.labels[zone_label]
    gateway_cluster = gateway.node.labels[zone_label]
    lookback_seconds_ago = now - parameters.lookback
    traces = ctx.trace_service.get_traces_for_function(fn, lookback_seconds_ago, now, gateway_cluster)

    if len(traces) > 0:
        # filter for client zone
        traces = traces[traces['origin_zone'] == client_gateway_cluster]
        traces = traces[traces['client'] == client]
        # percentile of rtt over all traces
        duration_agg = np.percentile(q=parameters.percentile_duration,
                                     a=traces[parameters.target_time_measure])
        if parameters.target_time_measure == 'rtt':
            # convert to ms
            duration_agg *= 1000

        d_moved = d - (d * offset)

        p_lat_a_x_f = logistic_curve(duration_agg, a, b, c, d_moved) / a
    else:
        p_lat_a_x_f = 0

    return p_lat_a_x_f


def pressure_by_num_requests_for_gateway(client: str, fn: str, traces: pd.DataFrame) -> float:
    """
    Calculates the pressure based on the number of requests that were served by the gateway.
    We consider only requests from the given function.
    This will yield higher pressure values for the gateway in case the client is spawning more requests than others.
    Dataframe contains: 'function', 'origin_zone', 'client'
    """
    try:
        df = traces[traces['function'] == fn]
        if len(df) == 0:
            value = 0
        else:
            # traces are already filtered for gateway
            R_a_f = len(df)
            unique_client_zones = len(df['origin_zone'].unique())
            client_traces = df[df['client'] == client]
            client_zone = client_traces['origin_zone'].iloc[0]
            client_zone_traces = df[df['origin_zone'] == client_zone]

            R_a_x_f = len(client_zone_traces)
            if R_a_x_f == 0:
                value = 0
            else:
                util = (R_a_f / unique_client_zones)

                value_max = 0
                for zone in df['origin_zone'].unique():
                    zone_traces = df[df['origin_zone'] == zone]
                    R_tmp = len(zone_traces)
                    value_tmp = R_tmp / util
                    if value_max < value_tmp:
                        value_max = value_tmp

                weight = R_a_x_f
                value = weight / util
                value = value / value_max

        return value
    except KeyError:
        logger.error(
            f"Wanted to access not called function {fn} to calculate num_request pressure of client {client}"
        )
        return 0


def pressure_by_network_distance(client_replica_id: str, max_latency: float, ctx: PlatformContext,
                                 gateway: FunctionReplica) -> float:
    """
    Calculates the pressure by network distance.
    Intuitively, the pressure represents how much more feasible it is to call the function inside the region.
    I.e., consider a latency between client and node of 50 ms, and an average latency of 10ms.
    Hosting the application inside the factory may therefore reduce the latency 5 times.
    Before returning, the value is normalized with respect to the maximum latency in the network (i.e.,
    Australia - Europe)
    :param client_replica_id: the client which called the function
    :param max_latency: the maximum latency in the whole network
    :param ctx: the context
    :param gateway: the gateway to which the client sends requests
    :return: pressure by network distance
    """
    client_node_name = ctx.replica_service.get_function_replica_by_id(client_replica_id).node.name
    client_node = ctx.node_service.find(client_node_name)
    client_gateway_node = \
        ctx.replica_service.find_function_replicas_with_labels({pod_type_label: api_gateway_type_label}, node_labels={
            zone_label: client_node.labels[zone_label]})[0]
    client_gateway_node_name = client_gateway_node.node.name
    avg_d = calculate_average_internal_distance(gateway, ctx)

    if client_gateway_node_name == gateway.node.name:
        # in case both are in the same -> then use average internal distance
        d_x_a = avg_d
    else:
        # otherwise set client gateway as client
        # use distance between the two gateways
        d_x_a = ctx.network_service.get_latency(client_gateway_node_name,
                                                gateway.node.name)

    value = d_x_a / max_latency

    return value


def calculate_average_internal_distance(gateway: FunctionReplica, ctx: PlatformContext) -> float:
    gateway_node = ctx.node_service.find(gateway.node.name)
    zone = gateway_node.labels[zone_label]
    nodes = ctx.node_service.find_nodes_in_zone(zone)
    distances = list(
        map(lambda c: ctx.network_service.get_latency(gateway.node.name, c.name), nodes)
    )
    return np.mean(distances)


def pressure_latency_fulfillment(client_replica_id: str, required_latency: float, ctx: PlatformContext,
                                 gateway: FunctionReplica) -> float:
    """
    Calculates the pressure based on the fulfillment of the latency requirement.
    I.e., applications have different requirements and therefore the pressure must reflect this circumstance
    :param client_replica_id: the client that calls
    :param required_latency: the required latency of the function
    :param ctx:
    :param gateway:
    :param result:
    :return: Between 0 and 1 scaled pressure, the higher the value the higher the requirement violation
    """
    client_node = ctx.replica_service.get_function_replica_by_id(client_replica_id).node
    client_gateway_node = \
        ctx.replica_service.find_function_replicas_with_labels({pod_type_label: api_gateway_type_label}, node_labels={
            zone_label: client_node.labels[zone_label]})[0]
    client_gateway_node_name = client_gateway_node.nodeName

    if client_gateway_node_name == gateway.node.name:
        # in case both are in the same -> then use average internal distance

        d_x_a = calculate_average_internal_distance(gateway, ctx)
    else:
        # otherwise set client gateway as client
        d_x_a = ctx.network_service.get_latency(client_gateway_node_name,
                                                gateway.node.name)

    v_a_x_f = d_x_a / required_latency

    p_lat_a_x_f = v_a_x_f

    return p_lat_a_x_f


def pressure_latency_req_log(parameters: PressureFunctionParameters, client_replica_id: str, latency_requirement: float,
                             gateway: FunctionReplica, ctx: PlatformContext):
    a = parameters.a
    b = parameters.b
    c = parameters.c
    offset = parameters.offset

    client_node = ctx.replica_service.get_function_replica_by_id(client_replica_id)
    client_gateway_node = \
        ctx.replica_service.find_function_replicas_with_labels({pod_type_label: api_gateway_type_label}, node_labels={
            zone_label: client_node.labels[zone_label]})[0]
    client_gateway_node_name = client_gateway_node.node.name
    client_gateway_zone = client_node.labels[zone_label]

    if client_gateway_node_name == gateway.node.name:
        # in case both are in the same -> then use average internal distance

        d_x_a = calculate_average_internal_distance(gateway, ctx)
    else:
        # otherwise set client gateway as client
        d_x_a = ctx.network_service.get_latency(client_gateway_node_name,
                                                gateway.node.name)

    d = latency_requirement - (latency_requirement * offset)

    p_lat_a_x_f = logistic_curve(d_x_a, a, b, c, d) / a

    return p_lat_a_x_f


def pressure_cpu_usage(parameters: PressureFunctionParameters, fn: str, gateway: FunctionReplica, now: float,
                       ctx: PlatformContext) -> float:
    gateway_node = gateway.node
    zone = gateway_node.labels[zone_label]
    cpu_usages = []
    running_replicas = ctx.replica_service.find_function_replicas_with_labels(
        labels={
            function_label: fn,
        },
        node_labels={
            zone_label: zone
        },
        running=True
    )
    for replica in running_replicas:
        lookback_seconds_ago = now - parameters.lookback
        cpu = ctx.telemetry_service.get_replica_cpu(replica.replica_id, lookback_seconds_ago, now)
        if cpu is None or len(cpu) == 0:
            pass
        else:
            cpu_usages.append(cpu)
    if len(cpu_usages) > 0:
        df_running = pd.concat(cpu_usages)
    else:
        logger.info(f"no cpu usage values for zone {zone}")
        return 0

    # mean utilization over all running pods
    mean_per_pod = df_running.groupby('container_id').mean()
    # this utilization is relative to the requested resources -> pressure of cpu usage in contrast to request
    # if 1 => fully utilized
    cpu_mean = mean_per_pod['percentage'].mean() / 100
    return cpu_mean


def get_scale_down_actions(pressure_values: pd.DataFrame, ctx: PlatformContext,
                           parameters: Dict[str, PressureAutoscalerParameters]) -> List[ScaleScheduleEvent]:
    """
     Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
     """
    not_under_pressure = is_not_under_pressure(pressure_values, ctx, parameters)
    logger.info("not under pressure_gateway %d", len(not_under_pressure))
    result = teardown_policy(ctx, not_under_pressure)
    return result


def get_under_pressure(pressure_values: pd.DataFrame, teardowns: int, ctx: PlatformContext,
                       parameters: Dict[str, PressureAutoscalerParameters]) -> List[
    PressureResult]:
    """
    Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
    """
    under_pressures = is_under_pressure(pressure_values, ctx, parameters)
    logger.info(" under pressure_gateway %d", len(under_pressures))
    under_pressure_new = []
    new_pods = {}
    for under_pressure in under_pressures:
        deployment = under_pressure.deployment
        max_replica = deployment.scaling_configuration.scale_max
        running_pods = len(ctx.replica_service.get_function_replicas_of_deployment(deployment.name))
        pending_pods = len(
            ctx.replica_service.get_function_replicas_of_deployment(deployment.name, running=False,
                                                                    state=pod_pending))
        all_pods = running_pods + pending_pods
        no_new_pods = new_pods.get(deployment.name, 0)
        if ((all_pods + no_new_pods) - teardowns) < max_replica:
            under_pressure_new.append(under_pressure)
            if new_pods.get(deployment.name, None) is None:
                new_pods[deployment.name] = 1
            else:
                new_pods[deployment.name] += 1

    return under_pressure_new


def is_under_pressure(pressure_per_gateway: pd.DataFrame, ctx: PlatformContext,
                      parameters: Dict[str, PressureAutoscalerParameters]) -> List[
    PressureResult]:
    """
   Checks for each gateway and deployment if its current pressure violates the threshold
   :return: gateway and deployment tuples that are under pressure
   """
    under_pressure = []
    if len(pressure_per_gateway) == 0:
        logger.info("No pressure values")
        return []
    # reduce df to look at the mean pressure over all clients per  function and zone
    for gateway in ctx.replica_service.find_function_replicas_with_labels(
            {pod_type_label: api_gateway_type_label}):
        gateway_node = gateway.node
        zone = gateway_node.labels[zone_label]
        for deployment in ctx.deployment_service.get_deployments():
            for client_zone in ctx.zone_service.get_zones():
                # client zone = x
                # check if pressure from x on a is too high, if yes -> try to schedule instance in x!
                try:
                    mean_pressure = pressure_per_gateway.loc[deployment.name]
                    if len(mean_pressure) == 0 or len(mean_pressure.loc[zone]) == 0 or len(
                            mean_pressure.loc[zone].loc[client_zone]) == 0:
                        continue
                    mean_pressure = mean_pressure.loc[zone].loc[client_zone][
                        'pressure']
                    if mean_pressure > parameters[zone].function_parameters[deployment.name].max_threshold:
                        # at this point we know where the origin for the high pressure comes from
                        target_gateway = ctx.replica_service.find_function_replicas_with_labels(
                            labels={pod_type_label: api_gateway_type_label},
                            node_labels={zone_label: client_zone})[0]
                        under_pressure.append(PressureResult(target_gateway, gateway, deployment))
                except KeyError:
                    pass
    return under_pressure


def is_not_under_pressure(pressure_values: pd.DataFrame, ctx: PlatformContext,
                          parameters: Dict[str, PressureAutoscalerParameters]) -> List[
    Tuple[FunctionReplica, FunctionDeployment]]:
    """
    Checks for each gateway if its current pressure violates the threshold
    Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
    :return: gateways that are under pressure
    """

    not_under_pressure = []
    # reduce df to look at the mean pressure over all clients per  function and zone
    if len(pressure_values) == 0:
        logger.info("No pressure values")
        return []

    pressure_df = pressure_values.groupby(['fn', 'fn_zone']).mean()
    for gateway in ctx.replica_service.find_function_replicas_with_labels(
            {pod_type_label: api_gateway_type_label}):
        gateway_node = gateway.node
        zone = gateway_node.labels[zone_label]
        for deployment in ctx.deployment_service.get_deployments():
            try:
                mean_pressure = pressure_df.loc[deployment.name].loc[zone]['pressure']
                if len(pressure_df.loc[deployment.name]) == 0 or len(
                        pressure_df.loc[deployment.name].loc[zone]) == 0:
                    continue
                if mean_pressure < parameters[zone].function_parameters[deployment.name].min_threshold:
                    pending_pods = ctx.replica_service.find_function_replicas_with_labels(
                        labels={
                            function_label: deployment.fn_name,
                        },
                        node_labels={
                            zone_label: zone
                        },
                        running=False,
                        state=pod_pending
                    )
                    if len(pending_pods) > 0:
                        logger.info(
                            f"Wanted to scale down FN {deployment.name} in zone {zone}, but had pending pods.")
                    else:
                        not_under_pressure.append((gateway, deployment))
            except KeyError:
                if deployment.labels.get(function_label, None) is not None:
                    logger.info(f'No pressure values found for {zone} - {deployment} - try to shut down')
                    not_under_pressure.append((gateway, deployment))
    return not_under_pressure


def teardown_policy(
        self,
        ctx: PlatformContext,
        scale_functions: List[Tuple[FunctionReplica, FunctionDeployment]],
        now: Callable[[], float]
) -> List[ScaleScheduleEvent]:
    """
    Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
    """
    scale_schedule_events = []
    backup = {}
    for event in scale_functions:
        deployment = event[1]
        fn = deployment.labels[function_label]
        all_replicas = ctx.replica_service.find_function_replicas_with_labels(
            {function_label: fn})
        backup[fn] = len(all_replicas)

    for event in scale_functions:
        gateway = event[0]
        deployment = event[1]
        fn = deployment.labels[function_label]
        gateway_node = ctx.node_service.find(gateway.node.name)
        zone = gateway_node.labels[zone_label]
        replicas = ctx.replica_service.find_function_replicas_with_labels(
            {function_label: fn}, node_labels={zone_label: zone})
        all_replicas = ctx.replica_service.find_function_replicas_with_labels(
            {function_label: fn})
        if len(replicas) == 0:
            logger.info(
                f"Wanted to remove container with function {fn} but there is no running container anymore in zone {zone}")
            continue
        if len(all_replicas) == 1:
            logger.info(
                f"Wanted to remove container with function {fn} but there is only one running container anymore in zone {zone}")
            continue
        if backup[fn] <= 1:
            logger.info(f"Tear down policy wanted to scale down function too many times")
            continue
        remove = self.replica_with_lowest_resource_usage(replicas, ctx)
        if remove is None:
            continue

        backup[fn] -= 1

        ts = now()

        event = ScaleScheduleEvent(
            ts=ts,
            fn=fn,
            replica=remove,
            origin_zone=zone,
            target_zone=zone,
            delete=True
        )
        scale_schedule_events.append(event)
    return scale_schedule_events
