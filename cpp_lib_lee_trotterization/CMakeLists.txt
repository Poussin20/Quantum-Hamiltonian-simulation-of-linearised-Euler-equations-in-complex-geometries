cmake_minimum_required(VERSION 3.10)
project(lee_trotter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

#here put the qulacs paths
include_directories(qulacs/include)
link_directories(qulacs/lib)      

pybind11_add_module(lee_trotter lee_trotter.cpp)

target_link_libraries(lee_trotter
    PRIVATE
        cppsim_static
        vqcsim_static
        csim_static
        OpenMP::OpenMP_CXX
)

#here put the location of this git before /numerical_comparaison
install(TARGETS lee_trotter
        LIBRARY DESTINATION /numerical_comparaison)