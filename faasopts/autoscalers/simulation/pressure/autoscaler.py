from faasopts.autoscalers.base.pressure.autoscaler import PressureAutoscaler


class SimPressureAutoscaler(PressureAutoscaler):

    def run(self):
        pressure_values = self.run()
        return pressure_values
