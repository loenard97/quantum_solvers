#pragma once

#include "_Definitions.hpp"

Vector bloch_rotation(const Vector& r, const Vector& Omega, double dt) {
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

inline void apply_propagator(Matrix& rho, const Matrix& U) {
    rho = U * rho * U.adjoint();
}

inline Matrix exp_propagator(const Vector& Omega, const double dt) {
    Matrix H = 0.5 * (Omega(0) * SIGMA_X + Omega(1) * SIGMA_Y + Omega(2) * SIGMA_Z);
    Matrix U = (-j * dt * H).exp();

    return U;
}

inline Matrix su2_propagator(const Vector& Omega, const double dt) {
    double omega = Omega.norm();

    if (omega < ZERO_CUTOFF) {
        return I;
    }

    Vector n = Omega / omega;
    double theta = omega * dt;
    double theta_cos = std::cos(theta / 2.0);
    double theta_sin = std::sin(theta / 2.0);

    Matrix U;
    Complex a = theta_cos - j * theta_sin * n(2);
    Complex b = -j * theta_sin * (n(0) - j*n(1));
    Complex c = -j * theta_sin * (n(0) + j*n(1));
    Complex d = theta_cos + j * theta_sin * n(2);
    U << a, b, c, d;

    return U;
}

inline Matrix cayley_propagator(const Vector& Omega, const double dt) {
    double omega = Omega.norm();

    if (omega < ZERO_CUTOFF) {
        return I;
    }

    double alpha = 0.25 * dt * omega;
    double alpha_2 = alpha * alpha;
    Vector n = Omega / omega;

    return (1.0-alpha_2)/(1.0+alpha_2) * I - (2.0*j*alpha)/(1.0+alpha_2) * (n(0) * SIGMA_X + n(1) * SIGMA_Y + n(2) * SIGMA_Z);
}