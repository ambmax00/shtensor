# include(ExternalProject)
include(FetchContent)

# ExternalProject_Add(ShtensorExt_fmt 
#                     PREFIX ${CMAKE_CURRENT_BINARY_DIR}/fmt
#                     GIT_REPOSITORY https://github.com/fmtlib/fmt.git
#                     CMAKE_ARGS -DFMT_TEST=FALSE 
#                                -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/fmt/install
# )

# set(SHTENSOR_FMT_LIB ${CMAKE_CURRENT_BINARY_DIR}/fmt/install/lib/libfmt.a)
# set(SHTENSOR_FMT_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/fmt/install/include)

# add_library(Shtensor_fmt STATIC IMPORTED GLOBAL DEPENDS ShtensorExt_fmt)
# set_target_properties(Shtensor_fmt PROPERTIES IMPORTED_LOCATION ${SHTENSOR_FMT_LIB})
# set_target_properties(Shtensor_fmt 
#                       PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${SHTENSOR_FMT_INCLUDE_DIR})

FetchContent_Declare(fmt
                     SYSTEM
                     GIT_REPOSITORY https://github.com/fmtlib/fmt.git 
)

FetchContent_Declare(asmjit
                     SYSTEM
                     GIT_REPOSITORY https://github.com/asmjit/asmjit.git)

FetchContent_MakeAvailable(fmt)
FetchContent_MakeAvailable(asmjit)

add_library(lapack_cpp INTERFACE)
target_include_directories(lapack_cpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lapack_cpp INTERFACE ${BLAS_LIBRARIES})