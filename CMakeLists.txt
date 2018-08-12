cmake_minimum_required(VERSION 3.9)
project(FTRL)

set(CMAKE_CXX_STANDARD 11)

add_executable(FTRL main.cpp src/model.cpp src/model.h src/utils.h src/dataset.cpp src/dataset.h src/constants.h)