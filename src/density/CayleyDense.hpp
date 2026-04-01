#include <complex>
#include <vector>
#include <tuple>

#include <Eigen/Dense>

#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "../Constants.hpp"

using Complex = std::complex<double>;
using Matrix = Eigen::Matrix2cd;
using Vector = Eigen::Vector3d;

class QubitCayleyDense {
public:
    double gamma_cooling = 0.0;
    double gamma_heating = 0.0;
    double gamma_dephasing = 0.0;
    Matrix rho;
    std::vector<Matrix> last_rhos;

    QubitCayleyDense() = default;

    /// @brief Return the last vector of density matrices as vector of bloch vector components
    /// @return 
    py::tuple as_bloch_vector()
    {
        size_t n = last_rhos.size();

        py::array_t<double> xs(n);
        py::array_t<double> ys(n);
        py::array_t<double> zs(n);

        auto xs_buf = xs.mutable_unchecked<1>();
        auto ys_buf = ys.mutable_unchecked<1>();
        auto zs_buf = zs.mutable_unchecked<1>();

        for (size_t i = 0; i < n; ++i) {
            xs_buf(i) = 2.0 * last_rhos[i](0, 1).real();
            ys_buf(i) = 2.0 * last_rhos[i](0, 1).imag();
            zs_buf(i) = (last_rhos[i](1, 1) - last_rhos[i](0, 0)).real();
        }

        return py::make_tuple(xs, ys, zs);
    }

    Matrix cayley_propagator(Vector Omega, double dt) {
        double omega = Omega.norm();

        if (omega < 1e-12) {
            return I;
        }

        double alpha = 0.25 * dt * omega;
        double alpha_2 = alpha * alpha;
        Vector n = Omega / omega;

        return (1-alpha_2) / (1+alpha_2) * I - (2*j*alpha) / (1+alpha_2) * (n(0) * SIGMA_X + n(1) * SIGMA_Y + n(2) * SIGMA_Z);
    }

    Matrix su2_propagator(Vector Omega, double dt) {
        double omega = Omega.norm();

        if (omega < 1e-12) {
            return I;
        }

        Vector n = Omega / omega;
        double theta = omega * dt;
        double theta_cos = std::cos(theta / 2.0);
        double theta_sin = std::sin(theta / 2.0);

        Matrix U;
        std::complex<double> a = theta_cos - j * theta_sin * n(2);
        std::complex<double> b = -j * theta_sin * (n(0) - j*n(1));
        std::complex<double> c = -j * theta_sin * (n(0) + j*n(1));
        std::complex<double> d = theta_cos + j * theta_sin * n(2);
        U << a, b, c, d;

        return U;
    }

    std::tuple<Matrix, Matrix> krauss_cooling(double tau) {
        double p;
        Matrix K0, K1;

        p = 1 - std::exp(-gamma_cooling * tau);
        K0 << 1.0, 0.0, 0.0, std::sqrt(1.0 - p);
        K1 << 0.0, std::sqrt(p), 0.0, 0.0;

        return std::tuple(K0, K1);
    }

    std::tuple<Matrix, Matrix> krauss_heating(double tau) {
        double p;
        Matrix K0, K1;

        p = 1 - std::exp(-gamma_heating * tau);
        K0 << std::sqrt(1.0 - p), 0.0, 0.0, 1.0;
        K1 << 0.0, 0.0, 0.0, std::sqrt(p);

        return std::tuple(K0, K1);
    }

    std::tuple<Matrix, Matrix> krauss_dephasing(double tau) {
        double p;
        Matrix K0, K1;

        p = 1 - std::exp(-gamma_dephasing * tau);
        K0 << std::sqrt(1.0 - 0.5*p), 0.0, 0.0, std::sqrt(1.0 - 0.5*p);
        K1 << std::sqrt(0.5*p), 0.0, 0.0, -std::sqrt(0.5*p);

        return std::tuple(K0, K1);
    }

    /// @brief Run exact solver for a single Hamiltonian
    /// @param Omega 
    /// @param rho 
    /// @param step 
    /// @param n_steps 
    /// @return Returns vector of density state matrices
    std::vector<Matrix> run_single(std::vector<Vector> Omega, double step) {
        Matrix U;
        std::vector<Matrix> rhos;
        std::tuple<Matrix, Matrix> Kc = krauss_cooling(0.5 * step);
        std::tuple<Matrix, Matrix> Kh = krauss_heating(0.5 * step);
        std::tuple<Matrix, Matrix> Kd = krauss_dephasing(0.5 * step);

        for (std::size_t i = 0; i < Omega.size(); ++i) {
            U = su2_propagator(Omega, step);

            // rho = std::get<0>(Kc) * rho * std::get<0>(Kc).adjoint() + std::get<1>(Kc) * rho * std::get<1>(Kc).adjoint();
            // rho = std::get<0>(Kh) * rho * std::get<0>(Kh).adjoint() + std::get<1>(Kh) * rho * std::get<1>(Kh).adjoint();
            // rho = std::get<0>(Kd) * rho * std::get<0>(Kd).adjoint() + std::get<1>(Kd) * rho * std::get<1>(Kd).adjoint();

            rho = U * rho * U.adjoint();
            
            // rho = std::get<0>(Kc) * rho * std::get<0>(Kc).adjoint() + std::get<1>(Kc) * rho * std::get<1>(Kc).adjoint();
            // rho = std::get<0>(Kh) * rho * std::get<0>(Kh).adjoint() + std::get<1>(Kh) * rho * std::get<1>(Kh).adjoint();
            // rho = std::get<0>(Kd) * rho * std::get<0>(Kd).adjoint() + std::get<1>(Kd) * rho * std::get<1>(Kd).adjoint();
            
            rhos.push_back(rho);
        }

        last_rhos = rhos;

        return rhos;
    }
    
    /// @brief Run exact solver for multiple Hamiltonians
    /// @param Omegas 
    /// @param rho 
    /// @param step 
    /// @param n_steps 
    /// @return 
    std::vector<std::vector<Matrix>> run_multiple(
        std::vector<Vector> Omegas, 
        double step, 
        size_t n_steps
    ) {
        std::vector<std::vector<Matrix>> states;
        std::vector<Matrix> rhos;
        
        for (size_t i = 0; i < Omegas.size(); ++i) {
            rhos = run_single(Omegas[i], step, n_steps);
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
    std::vector<Matrix> run_multiple_averaged(
        std::vector<Vector> Omegas, 
        double step, 
        size_t n_steps
    ) {
        Matrix cur_rho;
        std::vector<Matrix> mean_rhos(n_steps + 1, Matrix::Zero());
        mean_rhos[0] = rho;
        std::tuple<Matrix, Matrix> Kc;
        std::tuple<Matrix, Matrix> Kh;
        std::tuple<Matrix, Matrix> Kd;

        if (gamma_cooling < 1e-12 && gamma_heating < 1e-12 && gamma_dephasing < 1e-12) {
            for (size_t i = 0; i < Omegas.size(); ++i) {
                cur_rho = rho;
                Matrix U = su2_propagator(Omegas[i], step);

                for (size_t t = 1; t < n_steps + 1; ++t) {
                    cur_rho = U * cur_rho * U.adjoint();
                    mean_rhos[t] += (cur_rho - mean_rhos[t]) / double(i + 1);
                }
            }
        } else {
            Kc = krauss_cooling(0.5 * step);
            Kh = krauss_heating(0.5 * step);
            Kd = krauss_dephasing(0.5 * step);

            for (size_t i = 0; i < Omegas.size(); ++i) {
                cur_rho = rho;
                Matrix U = su2_propagator(Omegas[i], step);

                for (size_t t = 1; t < n_steps + 1; ++t) {
                    cur_rho = std::get<0>(Kc) * cur_rho * std::get<0>(Kc).adjoint() + std::get<1>(Kc) * cur_rho * std::get<1>(Kc).adjoint();
                    cur_rho = std::get<0>(Kh) * cur_rho * std::get<0>(Kh).adjoint() + std::get<1>(Kh) * cur_rho * std::get<1>(Kh).adjoint();
                    cur_rho = std::get<0>(Kd) * cur_rho * std::get<0>(Kd).adjoint() + std::get<1>(Kd) * cur_rho * std::get<1>(Kd).adjoint();

                    cur_rho = U * cur_rho * U.adjoint();
                    
                    cur_rho = std::get<0>(Kc) * cur_rho * std::get<0>(Kc).adjoint() + std::get<1>(Kc) * cur_rho * std::get<1>(Kc).adjoint();
                    cur_rho = std::get<0>(Kh) * cur_rho * std::get<0>(Kh).adjoint() + std::get<1>(Kh) * cur_rho * std::get<1>(Kh).adjoint();
                    cur_rho = std::get<0>(Kd) * cur_rho * std::get<0>(Kd).adjoint() + std::get<1>(Kd) * cur_rho * std::get<1>(Kd).adjoint();

                    mean_rhos[t] += (cur_rho - mean_rhos[t]) / double(i + 1);
                }
            }
        }

        rho = mean_rhos[n_steps + 1];
        last_rhos = mean_rhos;

        return mean_rhos;
    }
};