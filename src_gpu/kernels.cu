#pragma once

#include <cuda_runtime.h>

#include "CudaScalar.cuh"
#include "CudaMatrix.cuh"
#include "CudaVector.cuh"

#define ZERO_CUTOFF 1e-8

__host__ __device__
void apply_propagator(CudaMatrix& rho, CudaMatrix& U) {
    rho = U * rho * U.adj();
}

__host__ __device__
CudaMatrix su2_propagator(const CudaVector& Omega, const CudaScalar dt) {
    Complex j = Complex::j();
    CudaScalar omega = Omega.norm();

    if (omega < ZERO_CUTOFF) {
        return CudaMatrix::ones();
    }

    CudaVector n = Omega.normalized();
    CudaScalar theta = omega * dt;
    CudaScalar theta_cos = cosf(theta / 2.0);
    CudaScalar theta_sin = sinf(theta / 2.0);

    Complex a = Complex(theta_cos, 0.0) - j * theta_sin * n.z;
    Complex b = j * (-1.0) * theta_sin * (Complex(n.x, 0.0) - j*n.y);
    Complex c = j * (-1.0) * theta_sin * (Complex(n.x, 0.0) + j*n.y);
    Complex d = Complex(theta_cos, 0.0) + j * theta_sin * n.z;
    CudaMatrix U(a, b, c, d);

    return U;
}

__global__
void kernel(CudaMatrix* mean_data, int n, CudaMatrix rho_0, const CudaVector* Omegas, CudaScalar step, int n_steps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    CudaMatrix rho = rho_0;
    CudaMatrix U = su2_propagator(Omegas[i], step);

    for (int j = 0; j < n_steps; ++j) {
        apply_propagator(rho, U);
        atomicAdd(&mean_data[j].a00.r, rho.a00.r);
        atomicAdd(&mean_data[j].a00.i, rho.a00.i);
        atomicAdd(&mean_data[j].a01.r, rho.a01.r);
        atomicAdd(&mean_data[j].a01.i, rho.a01.i);
        atomicAdd(&mean_data[j].a10.r, rho.a10.r);
        atomicAdd(&mean_data[j].a10.i, rho.a10.i);
        atomicAdd(&mean_data[j].a11.r, rho.a11.r);
        atomicAdd(&mean_data[j].a11.i, rho.a11.i);
    }
}

__device__ inline void add_matrix(CudaMatrix& a, const CudaMatrix& b) {
    a.a00.r += b.a00.r; a.a00.i += b.a00.i;
    a.a01.r += b.a01.r; a.a01.i += b.a01.i;
    a.a10.r += b.a10.r; a.a10.i += b.a10.i;
    a.a11.r += b.a11.r; a.a11.i += b.a11.i;
}

__device__ inline void atomicAdd_matrix(CudaMatrix& a, const CudaMatrix& b) {
    atomicAdd(&a.a00.r, b.a00.r); atomicAdd(&a.a00.i, b.a00.i);
    atomicAdd(&a.a01.r, b.a01.r); atomicAdd(&a.a01.i, b.a01.i);
    atomicAdd(&a.a10.r, b.a10.r); atomicAdd(&a.a10.i, b.a10.i);
    atomicAdd(&a.a11.r, b.a11.r); atomicAdd(&a.a11.i, b.a11.i);
}

__global__
void kernel_mean_reduction(
    CudaMatrix* mean_data,   // global sum (not mean yet!)
    int n,
    CudaMatrix rho_0,
    const CudaVector* Omegas,
    CudaScalar step,
    int n_steps
) {
    extern __shared__ CudaMatrix shared[]; 
    // size = blockDim.x * n_steps

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Pointer to this thread's slice in shared memory
    CudaMatrix* local = &shared[tid * n_steps];

    // Initialize local sums
    for (int j = 0; j < n_steps; ++j) {
        // zero initialize
        local[j].a00 = {0,0};
        local[j].a01 = {0,0};
        local[j].a10 = {0,0};
        local[j].a11 = {0,0};
    }

    if (gid < n) {
        CudaMatrix rho = rho_0;
        CudaMatrix U = su2_propagator(Omegas[gid], step);

        for (int j = 0; j < n_steps; ++j) {
            apply_propagator(rho, U);
            add_matrix(local[j], rho);  // accumulate locally
        }
    }

    __syncthreads();

    // --- BLOCK REDUCTION (pairwise summation) ---
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            CudaMatrix* other = &shared[(tid + stride) * n_steps];
            for (int j = 0; j < n_steps; ++j) {
                add_matrix(local[j], other[j]);
            }
        }
        __syncthreads();
    }

    // --- WRITE BLOCK RESULT ---
    if (tid == 0) {
        CudaMatrix* block_sum = &shared[0];
        for (int j = 0; j < n_steps; ++j) {
            atomicAdd_matrix(mean_data[j], block_sum[j]);
        }
    }
}