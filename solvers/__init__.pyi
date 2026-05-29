import numpy.typing as npt


class Result:
    ts: npt.NDArray
    xs: npt.NDArray
    ys: npt.NDArray
    zs: npt.NDArray
    rhos: npt.NDArray

class Solver:
    def run(self, Omegas: float, step: float, n_steps: int) -> Result: ...
