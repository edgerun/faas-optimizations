from faasopts.autoscalers.base.pressure.osmotic import OsmoticAutoscaler


class SimOsmoticAutoscaler(OsmoticAutoscaler):

    def run(self):
        pressure_values = self.run()
        return pressure_values
