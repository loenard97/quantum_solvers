import time
import numpy as np
import matplotlib.pyplot as plt

import solvers


def single():
    solver = solvers.QubitExactDense()

    Omega = np.array([1.0, 0.0, 0.0])
    solver.rho = np.array([[1, 0], [0, 0]], dtype=complex)
    solver.gamma_cooling = 0.0
    solver.gamma_heating = 0.0
    solver.gamma_dephasing = 0.0

    T = 8*np.pi
    n_steps = 200
    step = T / n_steps
    ts = np.linspace(0, T, n_steps)
    
    print("run...")
    start = time.perf_counter()
    solver.run_single(Omega, step, n_steps)
    end = time.perf_counter()
    print(f"...done. Elapsed: {(end - start) * 1e6:.2f} us")
    xs, ys, zs = solver.as_bloch_vector()

    plt.figure()
    plt.plot(ts, xs)
    plt.plot(ts, ys)
    plt.plot(ts, zs)
    plt.show()


if __name__ == "__main__":
    single()
