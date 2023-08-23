set(SOURCES 
    Context.cpp 
    Kernel.cpp
    MemoryPool.cpp
    Tensor.cpp
    ThreadBarrier.cpp
    ThreadPool.cpp) 

add_library(shtensor ${SOURCES})

# mark asmjit include headers system to supress warnings
get_target_property(lib_include_dirs asmjit::asmjit INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(shtensor SYSTEM PRIVATE ${lib_include_dirs})

target_link_libraries(shtensor 
                      MPI::MPI_CXX 
                      fmt::fmt 
                      asmjit::asmjit
                      ${BLAS_LIBRARIES})