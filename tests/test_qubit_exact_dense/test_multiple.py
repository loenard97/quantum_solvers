import solvers
import numpy as np
import matplotlib.pyplot as plt


def multiple():
    qubit = solvers.SolverQubitExact()

    detunings = np.random.normal(loc=0, scale=0.5, size=1_000)
    Omegas = []
    for d in detunings:
        Omegas.append([1.0, 0.0, d])
    Omegas = np.array(Omegas)
    rho0 = np.array([[1, 0], [0, 0]], dtype=complex)

    T = 2*np.pi
    n_steps = 100
    step = T / n_steps
    print("run...")
    rhos = qubit.run_multiple(Omegas, rho0, step, n_steps)
    print("...done")

    ts = np.linspace(0, T, n_steps)
    xss, yss, zss = [], [], []
    for state in rhos:
        xs = [2*np.real(rho[0, 1]) for rho in state]
        ys = [2*np.imag(rho[0, 1]) for rho in state]
        zs = [np.real(rho[1, 1] - rho[0, 0]) for rho in state]
        xss.append(xs)
        yss.append(ys)
        zss.append(zs)
    xss = np.mean(np.array(xss), axis=0)
    yss = np.mean(np.array(yss), axis=0)
    zss = np.mean(np.array(zss), axis=0)

    plt.figure()
    plt.plot(ts, xss, label="x")
    plt.plot(ts, yss, label="y")
    plt.plot(ts, zss, label="z")
    plt.legend(loc="best", framealpha=0)
    plt.show()


if __name__ == "__main__":
    multiple()
