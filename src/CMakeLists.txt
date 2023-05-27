set(SOURCES 
    Context.cpp 
    MemoryPool.cpp
    Tensor.cpp
    ThreadBarrier.cpp
    ThreadPool.cpp) 

add_library(shtensor ${SOURCES})

target_link_libraries(shtensor MPI::MPI_CXX fmt::fmt)