# Installation

## Required Eigen3

Install with `vcpkg` on Windows
```
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
vcpkg install eigen3
```
and build with
```
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake
```

## Compile
Use this to compile to a `.pyd` file
```
g++ -O3 -shared -static -fPIC -I./pybind11/include -I./eigen -IC:\Users\denni\AppData\Local\Programs\Python\Python313\include $(python -m pybind11 --includes) .\src\_bindings.cpp -LC:\Users\denni\AppData\Local\Programs\Python\Python313\libs -lpython313 -o .\solvers\_core_windows.pyd
```
Compiling with `-ffast-math` actually made it slower. I think Eigen already optimizes so much, that compiler breaks things here.

On Linux
```
g++ -O3 -shared -fPIC -I/usr/include/pybind11 -I/usr/include/eigen3 $(python3-config --includes) $(python -m pybind11 --includes) src/_bindings.cpp $(python3-config --ldflags) -o solvers/_core_linux.so
```

## Install
Install as Python package with
```
pip install .
```


# Things yet to implement

## Installation with pip

```toml
[build-system]
requires = ["scikit-build-core", "pybind11"]
build-backend = "scikit_build_core.build"

[project]
name = "solver"
version = "0.1.0"
description = "Quantum solver with pybind11"
authors = [{name="You"}]
requires-python = ">=3.8"

[tool.scikit-build]
wheel.expand-macos-universal-tags = true
```

CMakeLists.txt
```txt
cmake_minimum_required(VERSION 3.15)
project(solver)

find_package(pybind11 REQUIRED)

pybind11_add_module(solver src/solver.cpp)

target_compile_features(solver PRIVATE cxx_std_17)
```

pip install -e . --no-build-isolation
