set(SOURCES 
    Context.cpp 
    MemoryPool.cpp
    Tensor.cpp) 

add_library(shtensor ${SOURCES})

target_link_libraries(shtensor MPI::MPI_CXX fmt::fmt)