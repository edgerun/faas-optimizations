from faasopts.autoscalers.base.pressure.autoscaler import PressureAutoscaler


class K8sPressureAutoscaler(PressureAutoscaler):

    def run(self):
        pressure_values = self.run()
        actions = self.find_local_scale_actions(pressure_values)
        for action in actions:
            if action.delete:
                self.scale_down(action.fn, action.replicas)
            else:
                self.scale_up(action.fn, action.replicas)

        return pressure_values
