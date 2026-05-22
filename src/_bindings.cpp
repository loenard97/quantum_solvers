// compile with
// g++ -shared -static -fPIC -I./pybind11/include -I./eigen -IC:\Users\denni\AppData\Local\Programs\Python\Python313\include $(python -m pybind11 --includes) solvers.cpp -LC:\Users\denni\AppData\Local\Programs\Python\Python313\libs -lpython313 -o solvers.pyd

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/complex.h>

namespace py = pybind11;

#include "EBCEND.hpp"
#include "EDCCKD.hpp"
#include "EDCCND.hpp"
#include "EDCEKD.hpp"
#include "EDCEND.hpp"
#include "EDTCKD.hpp"
#include "EDTCND.hpp"
#include "SBCEND.hpp"

#if defined(_WIN32) || defined(_WIN64)
    #define PYBIND_MODULE_NAME _core_windows
#elif defined(__linux__)
    #define PYBIND_MODULE_NAME _core_linux
#else
    #error "Unsupported platform"
#endif

PYBIND11_MODULE(PYBIND_MODULE_NAME, m_solvers) {
    py::class_<Result>(m_solvers, "Result")
        .def_readonly("rhos", &Result::rhos)
        .def(py::init<>())
        .def("__repr__", &Result::__repr)
        .def_property_readonly("ts", &Result::_get_ts)
        .def_property_readonly("xs", &Result::_get_xs)
        .def_property_readonly("ys", &Result::_get_ys)
        .def_property_readonly("zs", &Result::_get_zs);
    py::class_<EBCEND>(m_solvers, "EBCEND")
        .def_readwrite("r", &EBCEND::r)
        .def_readwrite("last_rs", &EBCEND::last_rs)
        .def(py::init<>())
        .def("run", &EBCEND::run);
    py::class_<EDCCKD>(m_solvers, "EDCCKD")
        .def_readwrite("gamma_cooling", &EDCCKD::gamma_cooling)
        .def_readwrite("gamma_heating", &EDCCKD::gamma_heating)
        .def_readwrite("gamma_dephasing", &EDCCKD::gamma_dephasing)
        .def_readwrite("rho", &EDCCKD::rho)
        .def_readwrite("last_rhos", &EDCCKD::last_rhos)
        .def(py::init<>())
        .def("as_bloch_vector", &EDCCKD::as_bloch_vector)
        .def("run", &EDCCKD::run);
    py::class_<EDCCND>(m_solvers, "EDCCND")
        .def_readwrite("rho", &EDCCND::rho)
        .def_readwrite("last_rhos", &EDCCND::last_rhos)
        .def(py::init<>())
        .def("as_bloch_vector", &EDCCND::as_bloch_vector)
        .def("run", &EDCCND::run);
    py::class_<EDCEKD>(m_solvers, "EDCEKD")
        .def_readwrite("gamma_cooling", &EDCEKD::gamma_cooling)
        .def_readwrite("gamma_heating", &EDCEKD::gamma_heating)
        .def_readwrite("gamma_dephasing", &EDCEKD::gamma_dephasing)
        .def_readwrite("rho", &EDCEKD::rho)
        .def(py::init<>())
        .def("run", &EDCEKD::run);
    py::class_<EDCEND>(m_solvers, "EDCEND")
        .def_readwrite("rho", &EDCEND::rho)
        .def(py::init<>())
        .def("run", &EDCEND::run);
    py::class_<EDTCKD>(m_solvers, "EDTCKD")
        .def_readwrite("gamma_cooling", &EDTCKD::gamma_cooling)
        .def_readwrite("gamma_heating", &EDTCKD::gamma_heating)
        .def_readwrite("gamma_dephasing", &EDTCKD::gamma_dephasing)
        .def_readwrite("rho", &EDTCKD::rho)
        .def_readwrite("last_rhos", &EDTCKD::last_rhos)
        .def(py::init<>())
        .def("as_bloch_vector", &EDTCKD::as_bloch_vector)
        .def("run", &EDTCKD::run);
    py::class_<EDTCND>(m_solvers, "EDTCND")
        .def_readwrite("rho", &EDTCND::rho)
        .def_readwrite("last_rhos", &EDTCND::last_rhos)
        .def(py::init<>())
        .def("as_bloch_vector", &EDTCND::as_bloch_vector)
        .def("run", &EDTCND::run);
    py::class_<SBCEND>(m_solvers, "SBCEND")
        .def_readwrite("r", &SBCEND::r)
        .def_readwrite("last_rs", &SBCEND::last_rs)
        .def(py::init<>())
        .def("run", &SBCEND::run);
}