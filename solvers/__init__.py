import sys
from typing import Literal

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib as mpl


if sys.platform.startswith("win"):
    from ._core_windows import *
    _CORE = _core_windows
elif sys.platform.startswith("linux"):
    from ._core_linux import *
    _CORE = _core_linux


class Result:

    ts: npt.NDArray
    xs: npt.NDArray
    ys: npt.NDArray
    zs: npt.NDArray
    rhos: npt.NDArray


class Solver:

    def __init__(
        self,
        system: Literal["Single", "Ensemble"],
        state: Literal["Bloch Vector", "Density Matrix"], 
        hamiltonian: Literal["Constant", "Time Dependent"], 
        propagator: Literal["Exact", "Cayley"],  # "Yoshida" 
        decay: Literal["None", "Krauss"],  # "Lindblad"
        output: Literal["Sparse", "Dense"],
    ):
        registry = {
            ("Ensemble", "Bloch Vector", "Constant", "Exact", "None", "Dense"): _CORE.EBCEND,
            ("Ensemble", "Density Matrix", "Constant", "Cayley", "Krauss", "Dense"): _CORE.EDCCKD,
            ("Ensemble", "Density Matrix", "Constant", "Cayley", "None", "Dense"): _CORE.EDCCND,
            ("Ensemble", "Density Matrix", "Time Dependent", "Cayley", "Krauss", "Dense"): _CORE.EDTCKD,
            ("Ensemble", "Density Matrix", "Time Dependent", "Cayley", "None", "Dense"): _CORE.EDTCND,
            ("Ensemble", "Density Matrix", "Constant", "Exact", "Krauss", "Dense"): _CORE.EDCEKD,
            ("Ensemble", "Density Matrix", "Constant", "Exact", "None", "Dense"): _CORE.EDCEND,
            ("Single", "Bloch Vector", "Constant", "Exact", "None", "Dense"): _CORE.SBCEND,
        }

        solver_cls = registry.get((system, state, hamiltonian, propagator, decay, output))
        if solver_cls is None:
            msg = "This combination of parameters is unavailable"
            raise ValueError(msg)
 
        self._solver = solver_cls()
        self._result = None

    @property
    def rho(self):
        return self._solver.rho

    @rho.setter
    def rho(self, value):
        self._solver.rho = value

    @property
    def result(self):
        return self._result

    def run(self, Omegas, step, n_steps) -> Result:
        self._result = self._solver.run(Omegas, step, n_steps)
        
        return self._result
    
    def plot_evolution(self):
        plt.figure()
        plt.plot(self._result.ts, self._result.xs, label="x")
        plt.plot(self._result.ts, self._result.ys, label="y")
        plt.plot(self._result.ts, self._result.zs, label="z")
        plt.legend(loc="best")
        plt.xlabel("time t")
        plt.ylabel("Bloch vector component")
        plt.show()


    def plot_evolution_bloch(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = np.array(self._result.xs)
        y = np.array(self._result.ys)
        z = -np.array(self._result.zs)

        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        t = np.linspace(0, 1, len(x))
        norm = mpl.colors.Normalize(0, 1)
        cmap = plt.cm.jet

        lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=2)
        lc.set_array(t)
        ax.add_collection3d(lc)

        u = np.linspace(0, 2*np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))

        ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.12, linewidth=0.5)

        L = 1.2
        ax.quiver(0, 0, 0, L, 0, 0, color="black", arrow_length_ratio=0.08, linewidth=1.5)
        ax.quiver(0, 0, 0, 0, L, 0, color="black", arrow_length_ratio=0.08, linewidth=1.5)
        ax.quiver(0, 0, 0, 0, 0, L, color="black", arrow_length_ratio=0.08, linewidth=1.5)
        ax.text(L, 0, 0, "X", fontsize=12)
        ax.text(0, L, 0, "Y", fontsize=12)
        ax.text(0, 0, L, "Z", fontsize=12)

        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])

        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(t)
        plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label="time")

        plt.show()
