from typing import Literal

from ._core_gpu_windows import *
_CORE = _core_gpu_windows


def Solver():
    return _CORE.GPU()
