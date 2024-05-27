import abc
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable

import numpy as np
import pandas as pd
from faas.context import PlatformContext
from faas.system import FunctionReplica, FunctionDeployment
from faas.util.constant import pod_type_label, api_gateway_type_label, zone_label, function_label, pod_pending, \
    worker_role_label

from faasopts.utils.pressure.api import PressureAutoscalerParameters, PressureFunctionParameters, \
    PressureScaleScheduleEvent

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
    scale_down: bool


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


def identify_above_max_pressure_deployments(pressure_values: pd.DataFrame, teardowns: Dict[str, int],
                                            ctx: PlatformContext,
                                            parameters: Dict[str, PressureAutoscalerParameters]) -> List[
    PressureResult]:
    """
    Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
    This function checks for pressure violations (i.e., higher than max threshold). Then, it checks
    which scale up actions (i.e., to mitigate the pressure violation) are feasible (i.e., keeping the number of replicas
    lower than the maximum amount allowed).
    This check also includes the replicas to teardown.
    Returns a list of results that indicate where a new replica of a given function should be spawned
    """
    above_max_pressure = is_above_max_threshold(pressure_values, ctx, parameters)
    logger.info(" under pressure_gateway %d", len(above_max_pressure))
    above_max_pressure_filtered = []
    new_pods = defaultdict(int)
    for under_pressure in above_max_pressure:
        deployment = under_pressure.deployment
        max_replica = deployment.scaling_configuration.scale_max
        running_pods = len(ctx.replica_service.get_function_replicas_of_deployment(deployment.name))
        pending_pods = len(
            ctx.replica_service.get_function_replicas_of_deployment(deployment.name, running=False,
                                                                    state=pod_pending))
        all_pods = running_pods + pending_pods
        no_new_pods = new_pods.get(deployment.name, 0)
        if ((all_pods + no_new_pods) - teardowns[deployment.name]) < max_replica:
            above_max_pressure_filtered.append(under_pressure)
            new_pods[deployment.name] += 1

    return above_max_pressure_filtered


def is_above_max_threshold(pressure_per_gateway: pd.DataFrame, ctx: PlatformContext,
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
                        under_pressure.append(PressureResult(target_gateway, gateway, deployment, False))
                except KeyError:
                    pass
    return under_pressure


def is_below_min_threshold(pressure_values: pd.DataFrame, ctx: PlatformContext,
                           parameters: Dict[str, PressureAutoscalerParameters]) -> List[PressureResult]:
    """
    Checks for each gateway if its current pressure violates the threshold
    Dataframe contains: 'fn', 'fn_zone', 'client_zone', 'pressure'
    :return: gateways that are below pressure
    """

    below_min_threshold = []
    if len(pressure_values) == 0:
        logger.info("No pressure values")
        return []

    # reduce df to look at the mean pressure over all clients per  function and zone
    # we only scale down if the incoming pressure on average is below the threshold
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
                        below_min_threshold.append(PressureResult(gateway, gateway, deployment, True))
            except KeyError:
                if deployment.labels.get(function_label, None) is not None:
                    logger.info(f'No pressure values found for {zone} - {deployment} - try to shut down')
                    below_min_threshold.append(PressureResult(gateway, gateway, deployment, True))
    return below_min_threshold


def teardown_policy(
        self,
        ctx: PlatformContext,
        scale_functions: List[Tuple[FunctionReplica, FunctionDeployment]],
        now: Callable[[], float]
) -> List[PressureScaleScheduleEvent]:
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

        event = PressureScaleScheduleEvent(
            ts=ts,
            fn=fn,
            replicas=[remove],
            origin_zone=zone,
            target_zone=zone,
            delete=True
        )
        scale_schedule_events.append(event)
    return scale_schedule_events


def prepare_pressure_scale_schedule_events(deployment: FunctionDeployment, new_target_zone: str,
                                          pressure_target_zone: str, local_scheduler_name: str, replica_factory,
                                           now: Callable[[], float], no_of_replicas) -> PressureScaleScheduleEvent:
    replicas = []
    for _ in range(no_of_replicas):
        replica = replica_factory.create_replica(
            {worker_role_label: 'true', 'origin_zone': pressure_target_zone,
             zone_label: new_target_zone, 'schedulerName': local_scheduler_name},
            deployment.deployment_ranking.get_first(), deployment)
        replicas.append(replica)

    time_time = now()

    scale_schedule_event = PressureScaleScheduleEvent(
        ts=time_time,
        fn=deployment.name,
        replicas=replicas,
        origin_zone=pressure_target_zone,
        target_zone=new_target_zone,
        delete=False
    )

    return scale_schedule_event


def is_zero_sum_action(create_events: Dict[str, List[str]], x: PressureScaleScheduleEvent):
    # see if the events have the same destination zone
    to_create = create_events.get(x.target_zone, None)
    if to_create is not None:

        # check if the function that is supposed to be deleted also in the list of containers to spawn
        if x.fn in to_create:
            return True

    return False


def remove_zero_sum_actions(create_results: List[PressureScaleScheduleEvent],
                            delete_results: List[PressureScaleScheduleEvent]):
    """
    Removes zero sum actions from create_results and delete_results
    :param create_results: tuples of replicas and target zones
    :param delete_results: delete actions to consider
    :return:
    """
    create_events = defaultdict(list)
    for result in create_results:
        create_events[result.target_zone].append(result.fn)

    # get all delete actions that are not reversing the creation event
    filtered_delete = list(
        filter(lambda x: not is_zero_sum_action(create_events, x), delete_results))
    logger.info(f"zero sum actions were identified: {len(filtered_delete) != delete_results}")
    return filtered_delete
