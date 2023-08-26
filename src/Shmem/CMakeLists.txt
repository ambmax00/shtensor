set(SOURCES 
    Context.cpp 
    MemoryPool.cpp) 

add_library(shtensor_shmem ${SOURCES})

target_include_directories(shtensor_shmem 
                           PUBLIC 
                           ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(shtensor_shmem
                      shtensor_basics
                      MPI::MPI_CXX)