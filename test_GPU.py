import solversgpu
import numpy as np
import matplotlib.pyplot as plt
import time


def multiple_averaged():
    solver = solversgpu.Solver()

    detunings = np.random.normal(loc=0, scale=0, size=100_000_000)
    print("n spins", len(detunings))
    # Omegas = np.array([[1.0, 0.0, d] for d in detunings])
    Omegas = np.empty((detunings.size, 3))
    Omegas[:, 0] = 1.0
    Omegas[:, 1] = 0.0
    Omegas[:, 2] = detunings
    T = 2*np.pi
    n_steps = 200
    step = T / n_steps

    print("run...")
    start = time.perf_counter()
    res = solver.run(Omegas, step, n_steps)
    end = time.perf_counter()
    print(f"...done. Elapsed: {(end - start):.2f} s")
    print(len(res.ts))

    plt.figure()
    plt.plot(res.ts, res.xs, label="x")
    plt.plot(res.ts, res.ys, label="y")
    plt.plot(res.ts, res.zs, label="z")
    plt.legend(loc="best", framealpha=0)
    plt.ylim(-1.1, 1.1)
    plt.show()


if __name__ == "__main__":
    multiple_averaged()
