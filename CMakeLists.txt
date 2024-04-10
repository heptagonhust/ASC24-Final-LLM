cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(trtllm-multinodes-instance LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
message(STATUS "Using MPI_CXX_INCLUDE_DIRS: ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "Using MPI_CXX_LIBRARIES: ${MPI_CXX_LIBRARIES}")

set(3RDPARTY_DIR "${PROJECT_SOURCE_DIR}/3rdparty")
add_subdirectory(
    "${3RDPARTY_DIR}/cxxopts/"
    "${3RDPARTY_DIR}/json/"
)

include_directories(
    ${CUDAToolkit_INCLUDE_DIRS} 
    ${MPI_CXX_INCLUDE_DIRS}
    ${3RDPARTY_DIR}/json/include
)

set(TRT_INCLUDE_DIR "/usr/local/tensorrt/include")
set(TRT_LIB_DIR "/usr/local/tensorrt/lib")
include_directories(${PROJECT_SOURCE_DIR}/include ${TRT_INCLUDE_DIR})

add_library(tensorrt_llm SHARED IMPORTED)
set(TRTLLM_LIB_LOCATION "${PROJECT_SOURCE_DIR}/libs/libtensorrt_llm.so")
set_property(TARGET tensorrt_llm PROPERTY IMPORTED_LOCATION ${TRTLLM_LIB_LOCATION})

add_library(libnvinfer_plugin_tensorrt_llm SHARED IMPORTED)
set(TRTLLM_PLUGIN_LIB_LOCATION "${PROJECT_SOURCE_DIR}/libs/libnvinfer_plugin_tensorrt_llm.so")
set_property(TARGET libnvinfer_plugin_tensorrt_llm PROPERTY IMPORTED_LOCATION 
                                                            ${TRTLLM_PLUGIN_LIB_LOCATION})

add_library(nvinfer SHARED IMPORTED)
set(NVINFER_LOCATION "${TRT_LIB_DIR}/libnvinfer.so")
set_property(TARGET nvinfer PROPERTY IMPORTED_LOCATION ${NVINFER_LOCATION})

add_library(libtokenizers_cpp SHARED IMPORTED)
set(LIBTOKENIZERS_CPP_LOCATION "${PROJECT_SOURCE_DIR}/libs/libtokenizers_cpp.a")
set_property(TARGET libtokenizers_cpp PROPERTY IMPORTED_LOCATION ${LIBTOKENIZERS_CPP_LOCATION})

add_library(libtokenizers_c SHARED IMPORTED)
set(LIBTOKENIZERS_C_LOCATION "${PROJECT_SOURCE_DIR}/libs/libtokenizers_c.a")
set_property(TARGET libtokenizers_c PROPERTY IMPORTED_LOCATION ${LIBTOKENIZERS_C_LOCATION})

add_library(librpc SHARED IMPORTED)
set(LIBRPC_LOCATION "${PROJECT_SOURCE_DIR}/libs/librpc.a")
set_property(TARGET librpc PROPERTY IMPORTED_LOCATION ${LIBRPC_LOCATION})

set(CLIENT_SHARED_TARGETS 
    tensorrt_llm 
    libnvinfer_plugin_tensorrt_llm
    nvinfer
    libtokenizers_cpp
    libtokenizers_c
    librpc)

set(SERVER_SHARED_TARGETS 
    libtokenizers_cpp
    libtokenizers_c
    librpc)


set(SERVER_INSTANCE_NAME server)
FILE(GLOB_RECURSE SOURCE_SERVER CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/cmmlu/*.cpp")
FILE(GLOB_RECURSE SOURCE_MMLU CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/mmlu/*.cpp")
add_executable(${SERVER_INSTANCE_NAME} ${SOURCE_SERVER} ${SOURCE_MMLU} "${PROJECT_SOURCE_DIR}/src/server/server.cpp")
add_definitions(-DENABLE_BF16)
add_definitions(-DENABLE_FP8)
target_compile_options(${SERVER_INSTANCE_NAME} PRIVATE -Wno-deprecated-declarations)
target_link_libraries(${SERVER_INSTANCE_NAME} PUBLIC 
                                        ${SERVER_SHARED_TARGETS}
                                        cxxopts::cxxopts
                                        )

set(CLIENT_INSTANCE_NAME client)
FILE(GLOB_RECURSE SOURCE_CLIENT CONFIGURE_DEPENDS 
    "${PROJECT_SOURCE_DIR}/src/model_instance/*.cpp")
add_executable(${CLIENT_INSTANCE_NAME} ${SOURCE_CLIENT} "${PROJECT_SOURCE_DIR}/src/client/client.cpp")
add_definitions(-DENABLE_BF16)
add_definitions(-DENABLE_FP8)
target_compile_options(${CLIENT_INSTANCE_NAME} PRIVATE -Wno-deprecated-declarations)
target_link_libraries(${CLIENT_INSTANCE_NAME} PUBLIC 
                                        ${CLIENT_SHARED_TARGETS}
                                        ${MPI_CXX_LIBRARIES}
                                        CUDA::cudart
                                        cxxopts::cxxopts
                                        )