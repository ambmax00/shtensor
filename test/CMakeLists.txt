add_executable(test_mempool test_mempool.cpp)
add_executable(test_tensor test_tensor.cpp)

include_directories(../src)

target_link_libraries(test_mempool shtensor)
target_link_libraries(test_tensor shtensor)

add_test(Test_mempool mpirun test_mempool)
add_test(Test_tensor mpirun test_tensor)