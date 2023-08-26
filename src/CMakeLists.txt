add_subdirectory(Basics)
add_subdirectory(Kernel)
add_subdirectory(Shmem)
add_subdirectory(Tensor)

add_library(shtensor INTERFACE)
target_link_libraries(shtensor 
                      INTERFACE 
                      shtensor_basics
                      shtensor_kernel
                      shtensor_shmem
                      shtensor_tensor)