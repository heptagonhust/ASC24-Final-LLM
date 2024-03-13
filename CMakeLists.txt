cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(trtllm-single-instance LANGUAGES CXX CUDA)
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


set(INSTANCE_NAME batched_inference)

add_library(tensorrt_llm SHARED IMPORTED)
set(TRTLLM_LIB_LOCATION "${PROJECT_SOURCE_DIR}/libs/libtensorrt_llm.so")
set_property(TARGET tensorrt_llm PROPERTY IMPORTED_LOCATION ${TRTLLM_LIB_LOCATION})

add_library(libnvinfer_plugin_tensorrt_llm SHARED IMPORTED)
set(TRTLLM_PLUGIN_LIB_LOCATION "${PROJECT_SOURCE_DIR}/libs/libnvinfer_plugin_tensorrt_llm.so")
set_property(TARGET libnvinfer_plugin_tensorrt_llm PROPERTY IMPORTED_LOCATION 
                                                            ${TRTLLM_PLUGIN_LIB_LOCATION})

add_library(libth_common SHARED IMPORTED)
set(LIBTH_LOCATION "${PROJECT_SOURCE_DIR}/libs/libth_common.so")
set_property(TARGET libth_common PROPERTY IMPORTED_LOCATION ${LIBTH_LOCATION})

add_library(nvinfer SHARED IMPORTED)
set(NVINFER_LOCATION "${TRT_LIB_DIR}/libnvinfer.so")
set_property(TARGET nvinfer PROPERTY IMPORTED_LOCATION ${NVINFER_LOCATION})


set(SHARED_TARGETS 
    tensorrt_llm 
    libnvinfer_plugin_tensorrt_llm
    libth_common
    nvinfer)

FILE(GLOB_RECURSE SOURCE CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_executable(${INSTANCE_NAME} ${SOURCE})
add_definitions(-DENABLE_BF16)
add_definitions(-DENABLE_FP8)
target_compile_options(${INSTANCE_NAME} PRIVATE -Wno-deprecated-declarations)
target_link_libraries(${INSTANCE_NAME} PUBLIC 
                                        ${SHARED_TARGETS}
                                        ${MPI_CXX_LIBRARIES}
                                        CUDA::cudart
                                        cxxopts::cxxopts
                                        )