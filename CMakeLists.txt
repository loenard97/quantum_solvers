cmake_minimum_required(VERSION 3.15)

project(solvers)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

pybind11_add_module(solvers MODULE src/solvers.cpp)
set_target_properties(solvers PROPERTIES
    OUTPUT_NAME "solvers"
    PREFIX ""
)

target_compile_features(solvers PRIVATE cxx_std_17)

target_link_libraries(solvers PRIVATE Eigen3::Eigen)