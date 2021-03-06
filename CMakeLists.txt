cmake_minimum_required(VERSION 2.8)
project(565proj2)

find_package(CUDA REQUIRED)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)

if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang|GNU")
    set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mtune=native -ffast-math")
    set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAG_DEBUG} -O0 -g -DDEBUG=1")
    list(APPEND CUDA_NVCC_FLAGS_DEBUG "-G")
endif()

list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)

add_library(prefix_sum
    "prefix_sum.h"
    "prefix_sum.cpp"
    )

cuda_add_library(compaction
    "cuda_helpers.h"
    "compaction.h"
    "compaction.cu"
    )

cuda_add_executable(main
    "main.cu"
    )
target_link_libraries(main prefix_sum compaction)
