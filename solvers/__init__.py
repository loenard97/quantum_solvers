import sys
from typing import Literal

if sys.platform.startswith("win"):
    from ._core_windows import *
    _CORE = _core_windows
elif sys.platform.startswith("linux"):
    from ._core_linux import *
    _CORE = _core_linux


def Solver(
    system: Literal["Single", "Ensemble"],
    state: Literal["Bloch Vector", "Density Matrix"], 
    hamiltonian: Literal["Constant", "Time Dependent"], 
    propagator: Literal["Exact", "Cayley"],  # "Yoshida" 
    decay: Literal["None", "Krauss"],  # "Lindblad"
    output: Literal["Sparse", "Dense"],
):
    register = {
        ("Ensemble", "Bloch Vector", "Constant", "Exact", "None", "Dense"): _CORE.EBCEND,
        ("Ensemble", "Density Matrix", "Constant", "Cayley", "Krauss", "Dense"): _CORE.EDCCKD,
        ("Ensemble", "Density Matrix", "Constant", "Cayley", "None", "Dense"): _CORE.EDCCND,
        ("Ensemble", "Density Matrix", "Time Dependent", "Cayley", "Krauss", "Dense"): _CORE.EDTCKD,
        ("Ensemble", "Density Matrix", "Time Dependent", "Cayley", "None", "Dense"): _CORE.EDTCND,
        ("Ensemble", "Density Matrix", "Constant", "Exact", "Krauss", "Dense"): _CORE.EDCEKD,
        ("Ensemble", "Density Matrix", "Constant", "Exact", "None", "Dense"): _CORE.EDCEND,
        ("Single", "Bloch Vector", "Constant", "Exact", "None", "Dense"): _CORE.SBCEND,
    }

    if hamiltonian == "Time Dependent" and propagator == "Exact":
        msg = "Time Dependent Hamiltonian is not available with Exact propagator"
        raise ValueError(msg)

    solver_cls = register.get((system, state, hamiltonian, propagator, decay, output))
    if solver_cls is None:
        msg = "This combination of parameters is unavailable"
        raise ValueError(msg)
    
    solver_instance = solver_cls()
    return solver_instance
