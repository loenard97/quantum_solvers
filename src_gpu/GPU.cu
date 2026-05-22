#pragma once

#include <iostream>
#include <cmath>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "CudaScalar.cuh"
#include "CudaComplex.cuh"
#include "CudaMatrix.cuh"
#include "CudaVector.cuh"
#include "kernels.cuh"

class Result {
public:
    py::array_t<CudaScalar> ts;
    py::array_t<CudaScalar> xs;
    py::array_t<CudaScalar> ys;
    py::array_t<CudaScalar> zs;

    Result(CudaMatrix rho_0, CudaMatrix* mean_data, CudaScalar step_size, size_t n_steps) {
        CudaScalar t = 0.0;

        ts = py::array_t<CudaScalar>(n_steps + 1);
        xs = py::array_t<CudaScalar>(n_steps + 1);
        ys = py::array_t<CudaScalar>(n_steps + 1);
        zs = py::array_t<CudaScalar>(n_steps + 1);
        auto buf_ts = ts.request();
        auto buf_xs = xs.request();
        auto buf_ys = ys.request();
        auto buf_zs = zs.request();
        CudaScalar* ptr_ts = (CudaScalar*) buf_ts.ptr;
        CudaScalar* ptr_xs = (CudaScalar*) buf_xs.ptr;
        CudaScalar* ptr_ys = (CudaScalar*) buf_ys.ptr;
        CudaScalar* ptr_zs = (CudaScalar*) buf_zs.ptr;

        ptr_ts[0] = t;
        ptr_xs[0] = 2.0 * rho_0.a01.r;
        ptr_ys[0] = 2.0 * rho_0.a01.i;
        ptr_zs[0] = rho_0.a11.r - rho_0.a00.r;
        for (int i = 1; i < n_steps + 1; ++i) {
            t += step_size;
            ptr_ts[i] = t;
            ptr_xs[i] = 2.0 * mean_data[i - 1].a01.r;
            ptr_ys[i] = 2.0 * mean_data[i - 1].a01.i;
            ptr_zs[i] = mean_data[i - 1].a11.r - mean_data[i - 1].a00.r;
        }
    }
};

class GPU {
public:
    __host__
    GPU() {}

    __host__
    Result run(py::array_t<CudaScalar> Omegas, CudaScalar step, size_t n_steps) {
        cudaError_t err;

        auto buf = Omegas.request();

        if (buf.ndim != 2 || buf.shape[1] != 3) {
            throw std::runtime_error("Omegas must be (N,3)");
        }

        int n = buf.shape[0];
        CudaScalar* ptr = static_cast<CudaScalar*>(buf.ptr);

        std::vector<CudaVector> host_omegas(n);

        for (int i = 0; i < n; ++i) {
            host_omegas[i] = CudaVector(
                ptr[3*i + 0],
                ptr[3*i + 1],
                ptr[3*i + 2]
            );
        }

        size_t n_omegas = host_omegas.size();
        int threads = 256;
        int blocks = (n_omegas + threads - 1) / threads;

        CudaMatrix rho_0(Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0));

        CudaVector* d_Omegas;
        cudaMalloc(&d_Omegas, n_omegas * sizeof(CudaVector));
        cudaMemcpy(d_Omegas, host_omegas.data(), n_omegas * sizeof(CudaVector), cudaMemcpyHostToDevice);
        CudaMatrix* mean_data = new CudaMatrix[n_steps];
        for (int i = 0; i < n_steps; ++i) {
            mean_data[i] = CudaMatrix::zeros();
        }
        CudaMatrix* d_mean;
        cudaMalloc(&d_mean, n_steps * sizeof(CudaMatrix));

        CudaMatrix *d_mean_data;
        cudaMalloc(&d_mean_data, n_steps * sizeof(CudaMatrix));
        cudaMemcpy(d_mean_data, mean_data, n_steps * sizeof(CudaMatrix), cudaMemcpyHostToDevice);

        /*
        size_t shm_size = threads * n_steps * sizeof(CudaMatrix);
        kernel_mean_reduction<<<blocks, threads, shm_size>>>(d_mean_data, n_omegas, rho_0, d_Omegas, step, n_steps);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA error for kernel_mean: " << cudaGetErrorString(err) << "\n"
            << blocks << ", " << threads << ", " << shm_size << std::endl;
        }

        for (int i = 0; i < n_steps; ++i) {
            mean_data[i] = mean_data[i] * (1.0f / n_omegas);
        }
        */

        kernel<<<blocks, threads>>>(d_mean_data, n_omegas, rho_0, d_Omegas, step, n_steps);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA error for kernel_mean: " << cudaGetErrorString(err) << std::endl;
        } 
        cudaDeviceSynchronize();
        cudaMemcpy(mean_data, d_mean_data, n_steps * sizeof(CudaMatrix), cudaMemcpyDeviceToHost);

        for (int i = 0; i < n_steps; ++i) {
            mean_data[i] = mean_data[i] * (1.0f / n_omegas);
        }

        Result result(rho_0, mean_data, step, n_steps);

        cudaFree(d_mean);
        cudaFree(d_Omegas);
        delete[] mean_data;

        return result;
    }
};