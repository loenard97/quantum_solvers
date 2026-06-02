from typing import Literal

import numpy.typing as npt


class Result:

    ts: npt.NDArray
    xs: npt.NDArray
    ys: npt.NDArray
    zs: npt.NDArray
    rhos: npt.NDArray


class Solver:

    rho: npt.NDArray

    def __init__(self,
        system: Literal["Single", "Ensemble"],
        state: Literal["Bloch Vector", "Density Matrix"], 
        hamiltonian: Literal["Constant", "Time Dependent"], 
        propagator: Literal["Exact", "Cayley"],
        decay: Literal["None", "Krauss"],
        output: Literal["Sparse", "Dense"]
    ) -> None: ...

    def run(self, Omegas: float, step: float, n_steps: int) -> Result: ...
