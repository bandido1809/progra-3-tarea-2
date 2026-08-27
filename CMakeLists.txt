cmake_minimum_required(VERSION 3.15)
project(TensorPP CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(tensorpp
        main.cpp
        Tensor.cpp
        Tensor.h)
