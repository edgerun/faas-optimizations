import abc
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from faas.context import PlatformContext
from faas.system import FunctionReplica
from faas.util.constant import pod_type_label, api_gateway_type_label, zone_label, function_label

from faasopts.autoscalers.base.pressure.autoscaler import PressureScalerParameters

logger = logging.getLogger(__name__)


@dataclass
class LogisticFunctionParameters:
    a: float  # max
    b: float  # bottom
    c: float  # growth
    d: float
    offset: float  # offset of midpoint (values > 0 increase the y-value of the midpoint)


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
    parameters: PressureScalerParameters
    client: str
    client_replica_id: str
    gateway: FunctionReplica
    function: str
    now: float
    traces: pd.DataFrame
    ctx: PlatformContext


class PressureFunction(abc.ABC):

    def __init__(self, weight: float):
        self.weight = weight

    def calculate_pressure(self, pressure_input: PressureInput) -> float:
        raise NotImplementedError()

    def calculate_weighted_pressure(self, pressure_input: PressureInput) -> float:
        return self.weight * self.calculate_pressure(pressure_input)

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
        return pressure_rtt_log(parameters, client, client_replica_id, gateway, fn, now, ctx)

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
        required_latency = parameters.function_requirements[fn]
        return pressure_latency_req_log(parameters, client_replica_id, required_latency, gateway, ctx)

    def name(self) -> str:
        return 'latency_req_log'


class PressureCpuUsageFunction(PressureFunction):

    def calculate_pressure(self, pressure_input: PressureInput) -> float:
        parameters = pressure_input.parameters
        now = pressure_input.now
        gateway = pressure_input.gateway
        fn = pressure_input.function
        ctx = pressure_input.ctx
        return pressure_cpu_usage(parameters, fn, gateway, now, ctx)

    def name(self) -> str:
        return 'cpu_usage'


def pressure_rtt_log(parameters: PressureScalerParameters, client: str, client_replica_id: str,
                     gateway: FunctionReplica, fn: str,
                     now: float, ctx: PlatformContext):
    a = parameters.logistic_function_parameters.a
    b = parameters.logistic_function_parameters.b
    c = parameters.logistic_function_parameters.c
    d = parameters.logistic_function_parameters.d
    offset = parameters.logistic_function_parameters.offset
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


def pressure_latency_req_log(parameters: PressureScalerParameters, client_replica_id: str, latency_requirement: float,
                             gateway: FunctionReplica, ctx: PlatformContext):
    a = parameters.logistic_function_parameters.a
    b = parameters.logistic_function_parameters.b
    c = parameters.logistic_function_parameters.c
    offset = parameters.logistic_function_parameters.offset

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


def pressure_cpu_usage(parameters: PressureScalerParameters, fn: str, gateway: FunctionReplica, now: float,
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
