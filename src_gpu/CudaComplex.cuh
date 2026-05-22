#pragma once

#include <cuda_runtime.h>

#include "CudaScalar.cuh"

struct Complex {
    CudaScalar r, i;

    __host__ __device__
    Complex(CudaScalar real = 0.0f, CudaScalar imag = 0.0f) : r(real), i(imag) {}

    __host__ __device__
    static Complex j() {
        return Complex(0.0, 1.0);
    }

    __host__ __device__
    Complex operator+(const Complex& b) const {
        return Complex(r + b.r, i + b.i);
    }

    __host__ __device__
    Complex operator-(const Complex& b) const {
        return Complex(r - b.r, i - b.i);
    }

    __host__ __device__
    Complex operator*(const Complex& b) const {
        return Complex(r * b.r - i * b.i, r * b.i + i * b.r);
    }

    __host__ __device__
    Complex conj() {
        return Complex(r, -i);
    }
};