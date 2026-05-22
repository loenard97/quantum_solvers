#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "CudaScalar.cuh"
#include "CudaComplex.cuh"

struct CudaMatrix {
    Complex a00, a01;
    Complex a10, a11;

    __host__ __device__
    CudaMatrix() {}

    __host__ __device__
    CudaMatrix(Complex m00, Complex m01,
           Complex m10, Complex m11)
        : a00(m00), a01(m01), a10(m10), a11(m11) {}

    __host__ __device__
    static CudaMatrix zeros() {
        return CudaMatrix(
            Complex(0, 0), Complex(0, 0),
            Complex(0, 0), Complex(0, 0)
        );
    }

    __host__ __device__
    static CudaMatrix ones() {
        return CudaMatrix(
            Complex(1, 0), Complex(0, 0),
            Complex(0, 0), Complex(1, 0)
        );
    }

    __host__
    void display() const {
        std::cout << "Mat2x2(\n"
            << "\t(" << a00.r << ", " << a00.i << ") (" << a01.r << ", " << a01.i << ")\n"
            << "\t(" << a10.r << ", " << a10.i << ") (" << a11.r << ", " << a11.i << ")\n"
            << ")\n";
    }

    __host__ __device__
    CudaMatrix operator+(const CudaMatrix& b) const {
        return CudaMatrix(
            a00 + b.a00, a01 + b.a01,
            a10 + b.a10, a11 + b.a11
        );
    }

    __host__ __device__
    CudaMatrix operator-(const CudaMatrix& b) const {
        return CudaMatrix(
            a00 - b.a00, a01 - b.a01,
            a10 - b.a10, a11 - b.a11
        );
    }

    __host__ __device__
    CudaMatrix operator*(CudaScalar s) const {
        return CudaMatrix(
            a00 * s, a01 * s,
            a10 * s, a11 * s
        );
    }

    __host__ __device__
    CudaMatrix operator*(const CudaMatrix& b) const {
        return CudaMatrix(
            a00 * b.a00 + a01 * b.a10,
            a00 * b.a01 + a01 * b.a11,
            a10 * b.a00 + a11 * b.a10,
            a10 * b.a01 + a11 * b.a11
        );
    }

    __host__ __device__
    CudaMatrix tran() {
        return CudaMatrix(a00, a10, a01, a11);
    }

    __host__ __device__
    CudaMatrix adj() {
        return CudaMatrix(a00.conj(), a10.conj(), a01.conj(), a11.conj());
    }
};