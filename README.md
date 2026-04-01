# Installation

## Required Eigen3

Install with `vcpkg` on Windows
```
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
vcpkg install eigen3
```
and build with
```
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake
```

## Compile
Use this to compile to a `.pyd` file
```
g++ -O3 -shared -static -fPIC -I./pybind11/include -I./eigen -IC:\Users\denni\AppData\Local\Programs\Python\Python313\include $(python -m pybind11 --includes) src/solvers.cpp -LC:\Users\denni\AppData\Local\Programs\Python\Python313\libs -lpython313 -o solvers.pyd
```
Compiling with `-ffast-math` actually made it slower. I think Eigen already optimizes so much, that compiler breaks things here.

## Install
Install as Python package with
```
pip install .
```


# Things yet to implement

## Installation with pip

```toml
[build-system]
requires = ["scikit-build-core", "pybind11"]
build-backend = "scikit_build_core.build"

[project]
name = "solver"
version = "0.1.0"
description = "Quantum solver with pybind11"
authors = [{name="You"}]
requires-python = ">=3.8"

[tool.scikit-build]
wheel.expand-macos-universal-tags = true
```

CMakeLists.txt
```txt
cmake_minimum_required(VERSION 3.15)
project(solver)

find_package(pybind11 REQUIRED)

pybind11_add_module(solver src/solver.cpp)

target_compile_features(solver PRIVATE cxx_std_17)
```

pip install -e . --no-build-isolation


## OpenMP parallization

```C++
#include <omp.h>

std::vector<Matrix> run_ensemble(
    const std::vector<Eigen::Vector3d>& Omegas,
    Matrix rho0,
    double step,
    size_t n_steps
) {
    size_t N = Omegas.size();

    // Global result
    std::vector<Matrix> mean_rhos(n_steps, Matrix::Zero());

    #pragma omp parallel
    {
        // Thread-local accumulator
        std::vector<Matrix> local_mean(n_steps, Matrix::Zero());

        #pragma omp for
        for (int i = 0; i < static_cast<int>(N); ++i) {
            Matrix rho = rho0;

            Matrix H = 0.5 * Omegas * SIGMA_X
                     + 0.5 * Omegas * SIGMA_Y
                     + 0.5 * Omegas * SIGMA_Z;

            Matrix U = (j * step * H).exp();

            for (size_t t = 0; t < n_steps; ++t) {
                rho = U * rho * U.adjoint();
                local_mean[t] += rho;  // accumulate locally
            }
        }

        // Combine into global result
        #pragma omp critical
        {
            for (size_t t = 0; t < n_steps; ++t) {
                mean_rhos[t] += local_mean[t];
            }
        }
    }

    // Final normalization
    for (size_t t = 0; t < n_steps; ++t) {
        mean_rhos[t] /= double(N);
    }

    return mean_rhos;
}
```

## SU(2) propagator
```C++
Matrix2cd su2_propagator(const Vector3d& Omega, double dt)
{
    double omega = Omega.norm();

    // Handle zero field (avoid division by zero)
    if (omega < 1e-12) {
        return Matrix2cd::Identity();
    }

    double theta = omega * dt;  // total angle
    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);

    Vector3d n = Omega / omega;

    Matrix2cd sigma_n =
        n(0) * SIGMA_X +
        n(1) * SIGMA_Y +
        n(2) * SIGMA_Z;

    return c * Matrix2cd::Identity() - I * s * sigma_n;
}
```

```C++
Matrix2cd U;

std::complex<double> a = c - I * s * n(2);
std::complex<double> b = -I * s * (n(0) - I*n(1));
std::complex<double> c = -I * s * (n(0) + I*n(1));
std::complex<double> d = c + I * s * n(2);

U << a, b, c, d;
```

## Bloch propagator
```C++
#include <Eigen/Dense>
#include <cmath>

using Vector3d = Eigen::Vector3d;

Vector3d rotate_bloch(const Vector3d& r,
                      const Vector3d& Omega,
                      double dt)
{
    double omega = Omega.norm();

    if (omega < 1e-12)
        return r;

    Vector3d n = Omega / omega;
    double theta = omega * dt;

    double c = std::cos(theta);
    double s = std::sin(theta);

    return r * c
         + n.cross(r) * s
         + n * (n.dot(r)) * (1.0 - c);
}
```