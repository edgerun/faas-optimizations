from typing import Callable

from faasopts.api import Optimizer


class ReconciliationOptimizationDaemon(Optimizer):

    def __init__(self, sleep: Callable[[],None], optimizer: Optimizer):
        self.sleep = sleep
        self.is_running = True
        self.optimizer = optimizer

    def setup(self):
        self.optimizer.setup()

    def run(self):
        while self.is_running:
            self.sleep()
            self.optimizer.run()

    def stop(self):
        self.optimizer.stop()
        self.is_running = False

