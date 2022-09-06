from typing import List, Union

from faas.opt.api import Optimizer
from faas.system import FunctionReplica


class BaseAutoscaler(Optimizer):
    def scale_down(self, function: str, remove: Union[int, List[FunctionReplica]]): ...

    def scale_up(self, function: str, add: Union[int, List[FunctionReplica]]): ...
