#!/bin/bash

GXX="g++"
GXX_FLAGS="-O3 -shared -fPIC"
GXX_INCLUDES="-I/usr/include/pybind11 -I/usr/include/eigen3"

PYTHON_INCLUDES=$(python3-config --includes)
PYTHON_FLAGS=$(python3-config --ldflags)
PYBIND11_INCLUDES=$(python -m pybind11 --includes)

INPUT_FILE="src/_bindings.cpp"
OUTPUT_FILE="solvers/_core_linux.so"

echo "compiling $OUTPUT_FILE"
$GXX $GXX_FLAGS $GXX_INCLUDES $PYTHON_INCLUDES $PYBIND11_INCLUDES $INPUT_FILE $PYTHON_FLAGS -o $OUTPUT_FILE