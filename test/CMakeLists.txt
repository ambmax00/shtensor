add_executable(test_mempool test_mempool.cpp)
add_executable(test_tensor test_tensor.cpp)
add_executable(test_window test_window.cpp)

include_directories(../src)

target_link_libraries(test_mempool shtensor)
target_link_libraries(test_tensor shtensor)
target_link_libraries(test_window shtensor)

message(STATUS "MPI executable: ${MPIEXEC_EXECUTABLE}")

add_test(Test_mempool ${MPIEXEC_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_mempool)
add_test(Test_tensor ${MPIEXEC_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_tensor)
add_test(Test_window ${MPIEXEC_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_window)