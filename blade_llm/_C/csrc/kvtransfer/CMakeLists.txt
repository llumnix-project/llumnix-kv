cmake_minimum_required(VERSION 3.16)
project(kvtransfer LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80 86 90)
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
option(BUILD_PYTHON_BIND "build interface expose to python" OFF)

find_package(CUDAToolkit)

file(GLOB_RECURSE SRCS CONFIGURE_DEPENDS src/*.cpp)
set(TARGET_INCLUDE_DIRS "/usr/local/include" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} include)
set(TARGET_LINK_LIBS CUDA::cudart)

if (BUILD_RDMA)
    set(TARGET_INCLUDE_DIRS ${TARGET_INCLUDE_DIRS} /usr/include/accl)
    set(TARGET_LINK_LIBS ${TARGET_LINK_LIBS} accl_barex)
endif ()

if (BUILD_PYTHON_BIND)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(kvtransfer_ops ${SRCS} kvtransfer_pybind.cpp)
    target_include_directories(kvtransfer_ops PRIVATE ${TARGET_INCLUDE_DIRS})
    target_link_libraries(kvtransfer_ops PUBLIC ${TARGET_LINK_LIBS})
else ()
    add_library(kvtransfer SHARED ${SRCS})
    target_compile_features(kvtransfer PUBLIC cxx_std_17)
    target_include_directories(kvtransfer PUBLIC ${TARGET_INCLUDE_DIRS})
    target_link_libraries(kvtransfer PUBLIC ${TARGET_LINK_LIBS})

    if (BUILD_TESTS)
        enable_testing()
        include(FetchContent)
        FetchContent_Declare(
                googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG        b514bdc898e2951020cbdca1304b75f5950d1f59 # release-1.15.2
        )
        FetchContent_MakeAvailable(googletest)
        file(GLOB_RECURSE TESTS CONFIGURE_DEPENDS tests/*.cpp tests/*.cu)
        add_executable(kv_transfer_test ${TESTS})
        target_compile_features(kv_transfer_test PUBLIC cxx_std_17)
        target_include_directories(kv_transfer_test PUBLIC ${TARGET_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS})
        set(TEST_LINK_LIBS ${TARGET_LINK_LIBS} ${GTEST_BOTH_LIBRARIES} GTest::gmock GTest::gmock_main kvtransfer)
        target_link_libraries(kv_transfer_test PUBLIC ${TEST_LINK_LIBS})
        set_target_properties(kv_transfer_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        include(GoogleTest)
        gtest_discover_tests(kv_transfer_test DISCOVERY_MODE PRE_TEST)
    endif ()
endif ()

