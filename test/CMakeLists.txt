add_executable(test0 test0.cpp)

include_directories(../src)

target_link_libraries(test0 shtensor)

add_test(Test0 mpirun test0)