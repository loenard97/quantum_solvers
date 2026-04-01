#include <complex>
#include <vector>

#include <Eigen/Dense>

#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "../Constants.hpp"

using Complex = std::complex<double>;
using Matrix = Eigen::Matrix2cd;
using Vector = Eigen::Vector3d;

class ExactDense {
public:
    Vector r;
    std::vector<Vector> last_rs;

    ExactDense() = default;

    Vector rotate_bloch(const Vector& r, const Vector& Omega, double dt) {
        double omega = Omega.norm();

        if (omega < 1e-12)
            return r;

        Vector n = Omega / omega;
        double theta = omega * dt;
        double theta_cos = std::cos(theta);
        double theta_sin = std::sin(theta);

        // I tried optimizing cross and dot product away, but that made it only slower
        return r * theta_cos + n.cross(r) * theta_sin + n * n.dot(r) * (1.0 - theta_cos);
    }

    /// @brief Run exact solver for a single Hamiltonian
    /// @param Omega 
    /// @param rho 
    /// @param step 
    /// @param n_steps 
    /// @return Returns vector of density state matrices
    std::vector<Vector> run_single(Vector Omega, double step, size_t n_steps) {
        std::vector<Vector> rs;

        for (std::size_t i = 0; i < n_steps; ++i) {
            r = rotate_bloch(r, Omega, step);
            rs.push_back(r);
        }

        last_rs = rs;

        return rs;
    }
    
    /// @brief Run exact solver for multiple Hamiltonians
    /// @param Omegas 
    /// @param rho 
    /// @param step 
    /// @param n_steps 
    /// @return 
    std::vector<std::vector<Vector>> run_multiple(std::vector<Vector> Omegas, double step, size_t n_steps) {
        std::vector<std::vector<Vector>> states;
        std::vector<Vector> rhos;
        
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
    std::vector<Vector> run_multiple_averaged(std::vector<Vector> Omegas, double step, size_t n_steps) {
        std::vector<Vector> mean_rs(n_steps, Vector::Zero());
        Vector cur_r;

        for (size_t i = 0; i < Omegas.size(); ++i) {
            cur_r = r;

            for (size_t t = 0; t < n_steps; ++t) {
                cur_r = rotate_bloch(cur_r, Omegas[i], step);
                mean_rs[t] += (cur_r - mean_rs[t]) / double(i + 1);
            }
        }

        r = mean_rs[n_steps];
        last_rs = mean_rs;

        return mean_rs;
    }
};