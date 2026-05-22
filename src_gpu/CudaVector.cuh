#pragma once

#include <cuda_runtime.h>

struct CudaVector {
    CudaScalar x, y, z;

    __host__ __device__
    CudaVector(CudaScalar x_=0.0f, CudaScalar y_=0.0f, CudaScalar z_=0.0f)
        : x(x_), y(y_), z(z_) {}

    __host__ __device__
    CudaVector operator+(const CudaVector& b) const {
        return CudaVector(x + b.x, y + b.y, z + b.z);
    }

    __host__ __device__
    CudaVector operator-(const CudaVector& b) const {
        return CudaVector(x - b.x, y - b.y, z - b.z);
    }

    __host__ __device__
    CudaVector operator*(CudaScalar s) const {
        return CudaVector(x * s, y * s, z * s);
    }

    __host__ __device__
    CudaScalar dot(const CudaVector& b) const {
        return x * b.x + y * b.y + z * b.z;
    }

    __host__ __device__
    CudaVector cross(const CudaVector& b) const {
        return CudaVector(
            y * b.z - z * b.y,
            z * b.x - x * b.z,
            x * b.y - y * b.x
        );
    }

    __host__ __device__
    CudaScalar norm() const {
        return sqrtf(x * x + y * y + z * z);
    }

    __host__ __device__
    CudaVector normalized() const {
        CudaScalar n = norm();
        return (n > 0.0f) ? (*this) * (1.0f / n) : CudaVector(0, 0, 0);
    }
};