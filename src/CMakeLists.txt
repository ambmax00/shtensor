set(SOURCES 
    Context.cpp 
    Tensor.cpp) 

add_library(shtensor ${SOURCES})

target_link_libraries(shtensor MPI::MPI_CXX)