#pragma once

#include <cuda_runtime.h>

#include "CudaScalar.cuh"
#include "CudaMatrix.cuh"
#include "CudaVector.cuh"

#define ZERO_CUTOFF 1e-8

__host__ __device__ void apply_propagator(CudaMatrix& rho, CudaMatrix& U);
__host__ __device__ CudaMatrix su2_propagator(const CudaVector& Omega, const CudaScalar dt);
__global__ void kernel(CudaMatrix* data, int n, CudaMatrix rho_0, const CudaVector* Omegas, CudaScalar step, int n_steps);
__global__ void kernel_mean_reduction(CudaMatrix* mean_data, int n, CudaMatrix rho_0, const CudaVector* Omegas, CudaScalar step, int n_steps);