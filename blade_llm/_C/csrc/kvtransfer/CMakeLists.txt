cmake_minimum_required(VERSION 3.16)
project(kvtransfer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -ggdb")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -ggdb")

option(BUILD_RDMA "build RDMA(accl-barex) transfer" ON)
if (BUILD_RDMA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_RDMA -DCMAKE_INCLUDE")
endif ()

option(BUILD_TESTS "build and run unit tests" OFF)
if (BUILD_TESTS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
endif ()
include_directories("/usr/local/include" "/usr/local/cuda-12.1/include")

find_package(Torch REQUIRED PATHS /usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch)
# if build test, use libtorch cxx11 ABI, and set Torch_DIR environment variable;
# find_package(Torch REQUIRED)
if (NOT TORCH_FOUND)
    message(FATAL_ERROR "libtoch library not found")
endif ()
find_package(PythonLibs REQUIRED)

file(GLOB_RECURSE SRCS CONFIGURE_DEPENDS src/*.cpp)
add_library(kvtransfer SHARED ${SRCS})
set_property(TARGET kvtransfer PROPERTY CXX_STANDARD 17)
set(TARGET_INCLUDE_DIRS include ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
set(TARGET_LINK_LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${PYTHON_LIBRARIES})

if (BUILD_RDMA)
    set(TARGET_INCLUDE_DIRS "${TARGET_INCLUDE_DIRS};/usr/include/accl")
    set(TARGET_LINK_LIBS "${TARGET_LINK_LIBS};accl_barex cuda")
endif ()

target_include_directories(kvtransfer PUBLIC ${TARGET_INCLUDE_DIRS})
target_link_libraries(kvtransfer PUBLIC ${TARGET_LINK_LIBS})

if (BUILD_TESTS)
    #    include(FetchContent)
    #    FetchContent_Declare(
    #            googletest
    #            GIT_REPOSITORY https://github.com/google/googletest.git
    #            GIT_TAG        703bd9caab50b139428cea1aaff9974ebee5742e # release-1.10.0
    #    )
    #    FetchContent_MakeAvailable(googletest)
    enable_testing()
    find_package(GTest REQUIRED PATHS /usr/local/lib/cmake/GTest)
    if(NOT GTEST_FOUND)
        message(FATAL_ERROR "GTest not found;")
    endif ()
    message(STATUS "include gtest from ${GTEST_FOUND}")
    file(GLOB_RECURSE TESTS CONFIGURE_DEPENDS tests/*.cpp tests/*.cu)
    add_executable(kv_transfer_test ${TESTS})
    target_include_directories(kv_transfer_test PUBLIC include ${GTEST_INCLUDE_DIRS})
    set(TEST_LINK_LIBS " ${GTEST_BOTH_LIBRARIES} GTest::gmock GTest::gmock_main kvtransfer")
    if (BUILD_RDMA)
        set(TEST_LINK_LIBS "${TEST_LINK_LIBS} accl_barex")
    endif ()
    target_link_libraries(kv_transfer_test PUBLIC ${TEST_LINK_LIBS})
    include(GoogleTest)
    gtest_discover_tests(kv_transfer_test DISCOVERY_MODE PRE_TEST)
endif ()