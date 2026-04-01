#pragma once

#include <complex>

#include <Eigen/Dense>

using Complex = std::complex<double>;
using Matrix = Eigen::Matrix2cd;

const Complex j = Complex(0, 1);
const Matrix SIGMA_X = (Matrix() << 0, 1, 1, 0).finished();
const Matrix SIGMA_Y = (Matrix() << 0, -j, j, 0).finished();
const Matrix SIGMA_Z = (Matrix() << 1, 0, 0, -1).finished();
const Matrix I = Matrix::Identity();