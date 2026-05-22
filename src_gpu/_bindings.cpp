// compile with
// nvcc -x cu -shared src_gpu/kernels.cu src_gpu/_bindings.cpp -I./pybind11/include -IC:\Users\denni\AppData\Local\Programs\Python\Python313\include -LC:\Users\denni\AppData\Local\Programs\Python\Python313\libs -lpython313 -o solversgpu\_core_gpu_windows.pyd

#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "GPU.cu"
#include "kernels.cuh"

#if defined(_WIN32) || defined(_WIN64)
    #define PYBIND_MODULE_NAME _core_gpu_windows
#elif defined(__linux__)
    #define PYBIND_MODULE_NAME _core_gpu_linux
#else
    #error "Unsupported platform"
#endif

PYBIND11_MODULE(PYBIND_MODULE_NAME, m_solvers) {
    py::class_<Result>(m_solvers, "Result")
        .def_readonly("ts", &Result::ts)
        .def_readonly("xs", &Result::xs)
        .def_readonly("ys", &Result::ys)
        .def_readonly("zs", &Result::zs);
    py::class_<GPU>(m_solvers, "GPU")
        .def(py::init<>())
        .def("run", &GPU::run);
}