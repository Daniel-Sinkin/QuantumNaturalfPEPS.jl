cmake_minimum_required(VERSION 3.18)
project(cuqnpeps LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(peps_sampler SHARED
  capi/capi.cu
  dtensor.cu
  dlenv/build.cu
  sampler/kernels.cu
  sampler/rangefinder.cu
  sampler/setup.cu
  sampler/sweep.cu
  sampler/entry.cu
)
target_include_directories(peps_sampler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(peps_sampler PROPERTIES
  OUTPUT_NAME peps_sampler
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
target_link_libraries(peps_sampler PRIVATE
  CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::curand
)
target_compile_options(peps_sampler PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wshadow,-Wconversion,-Wsign-conversion,-Wfloat-conversion,-Wcast-qual,-Wcast-align,-Wformat=2,-Wundef,-Wnull-dereference,-Wnon-virtual-dtor,-Woverloaded-virtual,-Wdouble-promotion,-Wimplicit-fallthrough,-Wreturn-type,-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=cuda>
)
target_link_options(peps_sampler PRIVATE -Wl,--no-undefined)
