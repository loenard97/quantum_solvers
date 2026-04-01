// compile with
// g++ -shared -static -fPIC -I./pybind11/include -I./eigen -IC:\Users\denni\AppData\Local\Programs\Python\Python313\include $(python -m pybind11 --includes) solvers.cpp -LC:\Users\denni\AppData\Local\Programs\Python\Python313\libs -lpython313 -o solvers.pyd

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/complex.h>

namespace py = pybind11;

#include "bloch/ExactDense.hpp"

#include "density/ExactDense.hpp"
#include "density/ExactSparse.hpp"

PYBIND11_MODULE(solvers, m_solvers) {
    py::module_ m_bloch = m_solvers.def_submodule("bloch");
    py::class_<ExactDense>(m_bloch, "ExactDense")
        .def_readwrite("r", &ExactDense::r)
        .def_readwrite("last_rs", &ExactDense::last_rs)
        .def(py::init<>())
        .def("run_single", &ExactDense::run_single)
        .def("run_multiple", &ExactDense::run_multiple)
        .def("run_multiple_averaged", &ExactDense::run_multiple_averaged);

    py::module_ m_density = m_solvers.def_submodule("density");
    py::class_<QubitExactDense>(m_density, "QubitExactDense")
        .def_readwrite("gamma_cooling", &QubitExactDense::gamma_cooling)
        .def_readwrite("gamma_heating", &QubitExactDense::gamma_heating)
        .def_readwrite("gamma_dephasing", &QubitExactDense::gamma_dephasing)
        .def_readwrite("rho", &QubitExactDense::rho)
        .def_readwrite("last_rhos", &QubitExactDense::last_rhos)
        .def(py::init<>())
        .def("as_bloch_vector", &QubitExactDense::as_bloch_vector)
        .def("run_single", &QubitExactDense::run_single)
        .def("run_multiple", &QubitExactDense::run_multiple)
        .def("run_multiple_averaged", &QubitExactDense::run_multiple_averaged);
    py::class_<QubitExactSparse>(m_density, "QubitExactSparse")
        .def_readwrite("rho", &QubitExactSparse::rho)
        .def(py::init<>())
        .def("as_bloch_vector", &QubitExactSparse::as_bloch_vector)
        .def("run_single", &QubitExactSparse::run_single)
        .def("run_multiple", &QubitExactSparse::run_multiple)
        .def("run_multiple_averaged", &QubitExactSparse::run_multiple_averaged);
}