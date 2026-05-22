#pragma once

#include <complex>
#include <vector>
#include <tuple>
#include <thread>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <pybind11/pybind11.h>

namespace py = pybind11;

// When a double falls below this value, it is assumed to be zero. This is used to guard against divisions by zero and 
// to avoid some computations. 
#define ZERO_CUTOFF 1e-12

using Complex = std::complex<double>;
using Matrix = Eigen::Matrix2cd;
using Vector = Eigen::Vector3d;

const Complex j = Complex(0, 1);
const Matrix SIGMA_X = (Matrix() << 0, 1, 1, 0).finished();
const Matrix SIGMA_Y = (Matrix() << 0, -j, j, 0).finished();
const Matrix SIGMA_Z = (Matrix() << 1, 0, 0, -1).finished();
const Matrix I = Matrix::Identity();

/// @brief Convert density matrices to bloch vectors
inline py::tuple _density_to_bloch(const std::vector<Matrix> rhos)
{
    size_t n = rhos.size();

    py::array_t<double> xs(n);
    py::array_t<double> ys(n);
    py::array_t<double> zs(n);

    auto xs_buf = xs.mutable_unchecked<1>();
    auto ys_buf = ys.mutable_unchecked<1>();
    auto zs_buf = zs.mutable_unchecked<1>();

    for (size_t i = 0; i < n; ++i) {
        xs_buf(i) = 2.0 * rhos[i](0, 1).real();
        ys_buf(i) = 2.0 * rhos[i](0, 1).imag();
        zs_buf(i) = (rhos[i](1, 1) - rhos[i](0, 0)).real();
    }

    return py::make_tuple(xs, ys, zs);
}

class Result {
public:
    std::vector<double> steps;
    std::vector<size_t> n_steps;
    Matrix rho_0;
    std::vector<Matrix> rhos;

    Result() = default;

    std::string __repr() {
        return "Result(" + std::to_string(rhos.size()) + " states)\n"
         + "\tts\ttimesteps\n"
         + "\txs\tBloch vector x\n"
         + "\tys\tBloch vector y\n"
         + "\tzs\tBloch vector z\n"
         + "\trhos\tDensity matrices";
    }

    std::vector<double> _get_ts() {
        std::vector<double> ts;

        double t = 0.0;
        for (size_t j = 0; j < n_steps.size(); ++j) {
            for (size_t i = 0; i < n_steps[j]; ++i) {
                t += steps[j];
                ts.push_back(t);
            }
        }

        return ts;
    }

    std::vector<double> _get_xs() {
        std::vector<double> xs;

        for (Matrix rho : rhos) {
            xs.push_back(2.0 * rho(0, 1).real());
        }

        return xs;
    }

    std::vector<double> _get_ys() {
        std::vector<double> ys;

        for (Matrix rho : rhos) {
            ys.push_back(2 * rho(0, 1).imag());
        }

        return ys;
    }

    std::vector<double> _get_zs() {
        std::vector<double> zs;

        for (Matrix rho : rhos) {
            zs.push_back((rho(1, 1) - rho(0, 0)).real());
        }

        return zs;
    }
};