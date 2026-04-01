from solvers.bloch import ExactDense
import numpy as np
import matplotlib.pyplot as plt
import time


def test():
    solver = ExactDense()
    solver.r = np.array([0, 0, -1])
    Omega = [1.0, 0.0, 0.0]

    T = 20*np.pi
    n_steps = 800
    step = T / n_steps
    print("run...")
    start = time.perf_counter()
    rs = solver.run_single(Omega, step, n_steps)
    end = time.perf_counter()
    print(f"...done. Elapsed: {(end - start):.2f} s")

    xs, ys, zs = np.array(rs).T
    ts = np.linspace(0, T, len(xs))

    plt.figure()
    plt.plot(ts, xs, label="x")
    plt.plot(ts, ys, label="y")
    plt.plot(ts, zs, label="z")
    plt.legend(loc="best", framealpha=0)
    plt.ylim(-1.1, 1.1)
    plt.show()


if __name__ == "__main__":
    test()
