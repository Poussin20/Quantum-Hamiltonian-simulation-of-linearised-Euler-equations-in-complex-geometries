cmake_minimum_required(VERSION 3.10)
project(lee_trotter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(/home/alexandre/OwnProject/QunaSys/qulacs/include)
link_directories(/home/alexandre/OwnProject/QunaSys/qulacs/lib)      

pybind11_add_module(lee_trotter lee_trotter.cpp)

target_link_libraries(lee_trotter
    PRIVATE
        cppsim_static
        vqcsim_static
        csim_static
        OpenMP::OpenMP_CXX
)

install(TARGETS lee_trotter
        LIBRARY DESTINATION /home/alexandre/OwnProject/QunaSys/QunaSys_assignment/numerical_comparaison)