set(SOURCES 
    AvxHelper.cpp
    Kernel.cpp
    KernelImpl.cpp) 

add_library(shtensor_kernel ${SOURCES})

# mark asmjit include headers system to supress warnings
get_target_property(lib_include_dirs asmjit::asmjit INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(shtensor_kernel SYSTEM PRIVATE ${lib_include_dirs})
target_include_directories(shtensor_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(shtensor_kernel 
                      shtensor_basics
                      asmjit::asmjit
                      lapack_cpp)