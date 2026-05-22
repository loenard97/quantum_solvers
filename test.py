from solvers.density import QubitExactDense
import numpy as np
import matplotlib.pyplot as plt
import time


def multiple_averaged():
    solver = QubitExactDense()
    solver.rho = np.array([[1, 0], [0, 0]], dtype=complex)
    detunings = np.random.normal(loc=0, scale=0.5, size=1_000)
    Omegas = [[1.0, 0.0, d] for d in detunings]

    T = 20*np.pi
    n_steps = 800
    step = T / n_steps
    ts = np.linspace(0, T, n_steps)
    print("run...")
    start = time.perf_counter()
    solver.run_multiple_averaged(Omegas, step, n_steps)
    end = time.perf_counter()
    print(f"...done. Elapsed: {(end - start):.2f} s")
    xs, ys, zs = solver.as_bloch_vector()

    plt.figure()
    plt.plot(ts, xs, label="x")
    plt.plot(ts, ys, label="y")
    plt.plot(ts, zs, label="z")
    plt.legend(loc="best", framealpha=0)
    plt.ylim(-1.1, 1.1)
    plt.show()


if __name__ == "__main__":
    multiple_averaged()
