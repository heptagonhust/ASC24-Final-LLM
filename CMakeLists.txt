cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(trtllm-single-instance LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

set(CXXOPTS_SRC_DIR "${PROJECT_SOURCE_DIR}/3rdparty/cxxopts/")
add_subdirectory(${CXXOPTS_SRC_DIR})

set(TRT_INCLUDE_DIR "/usr/local/tensorrt/include")
set(TRT_LIB_DIR "/usr/local/tensorrt/lib")
include_directories(${PROJECT_SOURCE_DIR}/include ${TRT_INCLUDE_DIR})

set(INSTANCE_NAME batched_inference)
set(CMAKE_CXX_STANDARD 17)

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
    nvinfer
    )

add_executable(${INSTANCE_NAME} src/main.cpp)
# target_compile_options(${INSTANCE_NAME} PRIVATE -w)
target_link_libraries(${INSTANCE_NAME} PUBLIC 
                                        ${SHARED_TARGETS}
                                        CUDA::cudart
                                        cxxopts::cxxopts)