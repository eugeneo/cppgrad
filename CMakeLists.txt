cmake_minimum_required(VERSION 3.20)

project(CppGrad VERSION 1.0)

set(CMAKE_CXX_STANDARD 20 )
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti -g -O3")

add_subdirectory(src)
add_subdirectory(test)