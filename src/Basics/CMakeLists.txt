set(SOURCES 
    ThreadBarrier.cpp
    ThreadPool.cpp
    Timer.cpp) 

add_library(shtensor_basics ${SOURCES})

target_include_directories(shtensor_basics
                           PUBLIC
                           ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(shtensor_basics 
                      PUBLIC
                      fmt::fmt 
)