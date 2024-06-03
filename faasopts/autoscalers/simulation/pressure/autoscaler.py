from typing import Callable

from faas.context import PlatformContext, FunctionReplicaFactory
from faas.system import Metrics, FaasSystem

from faasopts.autoscalers.base.pressure.autoscaler import PressureAutoscaler
from faasopts.utils.pressure.api import PressureAutoscalerParameters


class SimPressureAutoscaler(PressureAutoscaler):

    def __init__(self,  ctx: PlatformContext, parameters: PressureAutoscalerParameters,
                 zone: str,
                 replica_factory: FunctionReplicaFactory, now: Callable[[], float],
                 metrics: Metrics, result_q, faas: FaasSystem, env):
        super().__init__(ctx, parameters, zone, replica_factory, now, metrics)
        self.result_q = result_q
        self.faas = faas
        self.env = env

    def run(self):
        pressure_values = super().run()

        if pressure_values is not None:
            pressure_values = pressure_values.copy()
            pressure_values['solved'] = 0
            actions = self.find_local_scale_actions(pressure_values)
            for action in actions:
                if action.delete:
                    zone = action.origin_zone
                    fn = action.fn
                    # in case we scale down, we considered all incoming clients, therefore we set them all to solved
                    pressure_values.loc[fn].loc[zone].loc[:, 'solved'] = 1
                    self.env.process(self.faas.scale_down(action.fn, action.replicas))
                else:
                    zone = action.origin_zone
                    fn = action.fn
                    # in case of scale up, we only resolve internal scaling actions
                    pressure_values.loc[fn].loc[zone].loc[zone, 'solved'] = 1
                    self.env.process(self.faas.scale_up(action.fn, action.replicas))

            yield self.result_q.put(pressure_values)
        else:
            yield self.result_q.put(None)
