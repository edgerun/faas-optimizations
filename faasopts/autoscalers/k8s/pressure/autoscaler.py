from faasopts.autoscalers.base.pressure.autoscaler import PressureAutoscaler


class K8sPressureAutoscaler(PressureAutoscaler):

    def run(self):
        pressure_values = self.run()
        return pressure_values
