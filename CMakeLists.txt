cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(shtensor)

enable_language(CXX)
enable_testing()

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall -Wextra -pedantic)

find_package(MPI REQUIRED)

add_subdirectory(src)
add_subdirectory(test)
