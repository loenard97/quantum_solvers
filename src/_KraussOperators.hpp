#pragma once

#include "_Definitions.hpp"

inline void apply_krauss_operators(Matrix& rho, std::tuple<Matrix, Matrix>& Ks) {
    rho = std::get<0>(Ks) * rho * std::get<0>(Ks).adjoint() + std::get<1>(Ks) * rho * std::get<1>(Ks).adjoint();
}

inline std::tuple<Matrix, Matrix> krauss_cooling(const double gamma, const double tau) {
    double p;
    Matrix K0, K1;

    p = 1 - std::exp(-gamma * tau);
    K0 << 1.0, 0.0, 0.0, std::sqrt(1.0 - p);
    K1 << 0.0, std::sqrt(p), 0.0, 0.0;

    return std::tuple(K0, K1);
}

inline std::tuple<Matrix, Matrix> krauss_heating(const double gamma, const double tau) {
    double p;
    Matrix K0, K1;

    p = 1 - std::exp(-gamma * tau);
    K0 << std::sqrt(1.0 - p), 0.0, 0.0, 1.0;
    K1 << 0.0, 0.0, 0.0, std::sqrt(p);

    return std::tuple(K0, K1);
}

inline std::tuple<Matrix, Matrix> krauss_dephasing(const double gamma, const double tau) {
    double p;
    Matrix K0, K1;

    p = 1 - std::exp(-gamma * tau);
    K0 << std::sqrt(1.0 - 0.5*p), 0.0, 0.0, std::sqrt(1.0 - 0.5*p);
    K1 << std::sqrt(0.5*p), 0.0, 0.0, -std::sqrt(0.5*p);

    return std::tuple(K0, K1);
}