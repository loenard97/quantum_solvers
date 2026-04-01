#include <complex>
#include <vector>

#include <Eigen/Dense>

#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "../Constants.hpp"

using Complex = std::complex<double>;
using Matrix = Eigen::Matrix2cd;

class QubitExactSparse {
public:
    Matrix rho;

    QubitExactSparse() = default;

    /// @brief Return the last vector of density matrices as vector of bloch vector components
    /// @return 
    py::tuple as_bloch_vector()
    {
        double x, y, z;

        x = 2.0 * rho(0, 1).real();
        y = 2.0 * rho(0, 1).imag();
        z = (rho(1, 1) - rho(0, 0)).real();

        return py::make_tuple(x, y, z);
    }

    Matrix su2_propagator(Eigen::Vector3d Omega, double dt)
    {
        double omega = Omega.norm();

        // Handle zero field (avoid division by zero)
        if (omega < 1e-12) {
            return I;
        }

        double theta = omega * dt;
        double theta_cos = std::cos(theta / 2.0);
        double theta_sin = std::sin(theta / 2.0);

        Eigen::Vector3d n = Omega / omega;

        Matrix U;

        std::complex<double> a = theta_cos - j * theta_sin * n(2);
        std::complex<double> b = -j * theta_sin * (n(0) - j*n(1));
        std::complex<double> c = -j * theta_sin * (n(0) + j*n(1));
        std::complex<double> d = theta_cos + j * theta_sin * n(2);

        U << a, b, c, d;

        return U;
    }

    /// @brief Run exact solver for a single Hamiltonian
    /// @param Omega 
    /// @param rho 
    /// @param step 
    /// @param n_steps 
    /// @return Returns vector of density state matrices
    Matrix run_single(Eigen::Vector3d Omega, double step) {
        // Matrix H = 0.5 * Omega(0) * SIGMA_X + 0.5 * Omega(1) * SIGMA_Y + 0.5 * Omega(2) * SIGMA_Z;
        // Matrix U = (j * step * H).exp();
        Matrix U = su2_propagator(Omega, step);

        rho = U * rho * U.adjoint();

        return rho;
    }
    
    /// @brief Run exact solver for multiple Hamiltonians
    /// @param Omegas 
    /// @param rho 
    /// @param step 
    /// @param n_steps 
    /// @return 
    std::vector<Matrix> run_multiple(std::vector<Eigen::Vector3d> Omegas, double step) {
        std::vector<Matrix> states;
        Matrix rhos;
        
        for (size_t i = 0; i < Omegas.size(); ++i) {
            rhos = run_single(Omegas[i], step);
            states.push_back(rhos);
        }

        return states;
    }

    /// @brief Run exect solver for multiple Hamiltonians and average over all density matrices
    /// @param Omegas 
    /// @param rho0 
    /// @param step 
    /// @param n_steps 
    /// @return 
    Matrix run_multiple_averaged(std::vector<Eigen::Vector3d> Omegas, double step) {
        Matrix mean_rho = Matrix::Zero();
        Matrix cur_rho;

        for (size_t i = 0; i < Omegas.size(); ++i) {
            cur_rho = rho;
            Matrix H = 0.5 * Omegas[i](0) * SIGMA_X + 0.5 * Omegas[i](1) * SIGMA_Y + 0.5 * Omegas[i](2) * SIGMA_Z;
            // Matrix U = (-j * H * step).exp();
            Matrix U = su2_propagator(Omegas[i], step);

            cur_rho = U * cur_rho * U.adjoint();
            mean_rho += (cur_rho - mean_rho) / double(i + 1);
        }

        rho = mean_rho;

        return mean_rho;
    }
};