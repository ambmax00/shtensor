set(SOURCES 
    Tensor.cpp) 

add_library(shtensor_tensor ${SOURCES})

target_include_directories(shtensor_tensor 
                           PUBLIC 
                           ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(shtensor_tensor
                      shtensor_basics
                      shtensor_shmem)