import unittest
import numpy as np
import solvers


class TestEDCEKD(unittest.TestCase):

    def setUp(self):
        self.solver = solvers.Solver(
            system="Ensemble",
            state="Density Matrix",
            hamiltonian="Constant",
            propagator="Exact",
            decay="Krauss",
            output="Dense",
        )
    
    def test_run(self):
        self.solver.rho = np.array([[0, 0], [0, 1]], dtype=complex)
        Omegas = [[1.0, 0.0, 0.0]]

        T = 2*np.pi
        n_steps = 200
        step = T / n_steps

        res = self.solver.run(Omegas, step, n_steps)

        expected = {
            "xs": np.zeros(200), 
            "ys": -np.sin(res.ts), 
            "zs": np.cos(res.ts), 
        }

        np.testing.assert_allclose(res.xs, expected["xs"], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(res.ys, expected["ys"], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(res.zs, expected["zs"], rtol=1e-8, atol=1e-10)
