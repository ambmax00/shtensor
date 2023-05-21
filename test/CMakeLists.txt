include_directories(../src)

function(register_test name)

  add_executable(test_${name} test_${name}.cpp)
  target_link_libraries(test_${name} shtensor)
  add_test(Test_${name} ${MPIEXEC_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_${name})

endfunction()

message(STATUS "MPI executable: ${MPIEXEC_EXECUTABLE}")

register_test(mempool)
register_test(tensor)
register_test(window)
register_test(threadpool)