cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(trtllm-single-instance LANGUAGES CXX)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(INSTANCE_NAME batched_inference)
set(CMAKE_CXX_STANDARD 17)

add_library(tensorrt_llm SHARED IMPORTED)
set(TRT_LIB_LOCATION "${PROJECT_SOURCE_DIR}/libs/libtensorrt_llm.so")
set_property(TARGET tensorrt_llm PROPERTY IMPORTED_LOCATION ${TRT_LIB_LOCATION})

add_library(libnvinfer_plugin_tensorrt_llm SHARED IMPORTED)
set(TRT_PLUGIN_LIB_LOCATION "${PROJECT_SOURCE_DIR}/libs/libnvinfer_plugin_tensorrt_llm.so")
set_property(TARGET libnvinfer_plugin_tensorrt_llm PROPERTY IMPORTED_LOCATION 
                                                            ${TRT_PLUGIN_LIB_LOCATION})

set(CXXOPTS_SRC_DIR "${PROJECT_SOURCE_DIR}/3rdparty/cxxopts/")
add_subdirectory(${CXXOPTS_SRC_DIR})

set(SHARED_TARGETS 
    tensorrt_llm 
    libnvinfer_plugin_tensorrt_llm)

add_executable(${INSTANCE_NAME} src/main.cpp)
target_link_libraries(${INSTANCE_NAME} PUBLIC 
                                        ${SHARED_TARGETS}
                                        cxxopts::cxxopts)