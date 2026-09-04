cmake_minimum_required(VERSION 3.18)
project(cuqnpeps LANGUAGES CXX CUDA)

set(QNPEPS_C_API_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../c_api_version.txt")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${QNPEPS_C_API_VERSION_FILE}")
file(READ "${QNPEPS_C_API_VERSION_FILE}" QNPEPS_C_API_VERSION)
string(STRIP "${QNPEPS_C_API_VERSION}" QNPEPS_C_API_VERSION)
if(NOT QNPEPS_C_API_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR
    "c_api_version.txt must contain one major.minor.patch version, got: ${QNPEPS_C_API_VERSION}"
  )
endif()

set(QNPEPS_PROJECT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../Project.toml")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${QNPEPS_PROJECT_FILE}")
file(STRINGS "${QNPEPS_PROJECT_FILE}" QNPEPS_PROJECT_VERSION_LINES
  REGEX "^version = \"[0-9]+\\.[0-9]+\\.[0-9]+\"$"
)
list(LENGTH QNPEPS_PROJECT_VERSION_LINES QNPEPS_PROJECT_VERSION_COUNT)
if(NOT QNPEPS_PROJECT_VERSION_COUNT EQUAL 1)
  message(FATAL_ERROR "Project.toml must contain one major.minor.patch package version")
endif()
list(GET QNPEPS_PROJECT_VERSION_LINES 0 QNPEPS_PROJECT_VERSION_LINE)
string(REGEX REPLACE "^version = \"([^\"]+)\"$" "\\1"
  QNPEPS_PACKAGE_VERSION "${QNPEPS_PROJECT_VERSION_LINE}"
)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
include(GNUInstallDirs)

add_library(qnpeps SHARED
  capi/capi.cu
  tensor/contraction.cu
  core/error.cu
  tensor/tensor.cu
  tensor/kernels.cu
  tensor/permutation.cu
  peps/kernels.cu
  peps/init.cu
  densitymatrix/backend.cu
  densitymatrix/capi.cu
  densitymatrix/engine.cu
  densitymatrix/kernels.cu
  zipup/kernels.cu
  zipup/zipup_mpo_mps.cu
  gram/gram.cu
  gram/kernels.cu
  gram/gram_cublas.cu
  gram/gram_slab.cu
  minsr/adapter.cu
  minsr/kernels.cu
  minsr/solve.cu
  minsr/composition.cu
  minsr/distributed.cu
  eo/capi.cu
  eo/density.cu
  eo/eloc_kernels.cu
  eo/shards.cu
  linalg/eigensolve.cu
  linalg/factorization.cu
  linalg/gram.cu
  linalg/qr.cu
  linalg/svd.cu
  linalg/eo_ops.cu
  linalg/eo_routes.cu
  eo/dtensor.cu
  eo/env_build.cu
  e2e/capi.cu
  e2e/gram.cu
  e2e/kernels.cu
  e2e/update.cu
  e2e/step.cu
  e2e/node.cu
  e2e/node/workers.cu
  e2e/node/solve.cu
  e2e/node/lifecycle.cu
  e2e/node/step_api.cu
  e2e/node/compat.cu
  dlenv/build/execute.cu
  dlenv/build/zipup_context.cu
  dlenv/kernels.cu
  dlenv/density.cu
  sampler/kernels.cu
  sampler/density.cu
  sampler/rangefinder.cu
  sampler/setup.cu
  sampler/sweep/refresh.cu
  sampler/sweep/run.cu
  sampler/sweep_bridge.cu
  sampler/entry.cu
)
set_source_files_properties(
  eo/capi.cu
  eo/density.cu
  eo/eloc_kernels.cu
  eo/shards.cu
  linalg/eo_ops.cu
  linalg/eo_routes.cu
  eo/dtensor.cu
  eo/env_build.cu
  e2e/capi.cu
  e2e/gram.cu
  minsr/solve.cu
  minsr/composition.cu
  minsr/distributed.cu
  e2e/update.cu
  e2e/step.cu
  e2e/node.cu
  e2e/node/workers.cu
  e2e/node/solve.cu
  e2e/node/lifecycle.cu
  e2e/node/step_api.cu
  e2e/node/compat.cu
  PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-error>"
)
target_include_directories(qnpeps PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/eo
  ${CMAKE_CURRENT_SOURCE_DIR}/e2e
)
target_compile_definitions(qnpeps PRIVATE
  QNPEPS_C_API_VERSION="${QNPEPS_C_API_VERSION}"
  QNPEPS_PACKAGE_VERSION="${QNPEPS_PACKAGE_VERSION}"
)
set_target_properties(qnpeps PROPERTIES
  OUTPUT_NAME qnpeps
  PREFIX ""
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
target_link_libraries(qnpeps PRIVATE
  CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::curand Threads::Threads
)
target_compile_options(qnpeps PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wshadow,-Wconversion,-Wsign-conversion,-Wfloat-conversion,-Wcast-qual,-Wcast-align,-Wformat=2,-Wundef,-Wnull-dereference,-Wnon-virtual-dtor,-Woverloaded-virtual,-Wdouble-promotion,-Wimplicit-fallthrough,-Wreturn-type,-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=cuda>
)
target_link_options(qnpeps PRIVATE -Wl,--no-undefined)

add_executable(qnpeps_common_probe core/common_probe.cu)
target_include_directories(qnpeps_common_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(qnpeps_common_probe PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wshadow,-Wconversion,-Wsign-conversion,-Wfloat-conversion,-Wcast-qual,-Wcast-align,-Wformat=2,-Wnull-dereference,-Wnon-virtual-dtor,-Woverloaded-virtual,-Wdouble-promotion,-Wimplicit-fallthrough,-Wreturn-type,-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=cuda>
)

install(TARGETS qnpeps LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES
  capi/qnpeps.h
  eo/dans_qnpeps_eloc.h
  e2e/dans_qnpeps_e2e.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cuQuantumNaturalfPEPS
)

option(QNPEPS_BUILD_NATIVE_GATES "Build native CUDA correctness gates" OFF)
if(QNPEPS_BUILD_NATIVE_GATES)
  include(CTest)
  find_package(BLAS REQUIRED)
  get_filename_component(QNPEPS_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  set(QNPEPS_EO_ORACLE_DIR "${QNPEPS_REPO_ROOT}/attic/src_milestone3")
  set(QNPEPS_EO_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/eo-oracle")
  file(MAKE_DIRECTORY "${QNPEPS_EO_GENERATED_DIR}")

  foreach(source_name eloc hamiltonian)
    file(READ "${QNPEPS_EO_ORACLE_DIR}/${source_name}.cpp" source_text)
    string(REPLACE "{0.0, 0.0}" "{0.0}" source_text "${source_text}")
    file(WRITE "${QNPEPS_EO_GENERATED_DIR}/${source_name}.cpp" "${source_text}")
  endforeach()
  foreach(source_name ok minsr)
    file(READ "${QNPEPS_EO_ORACLE_DIR}/${source_name}.cpp" source_text)
    string(REPLACE "{0.0, 0.0}" "{0.0}" source_text "${source_text}")
    string(REPLACE "std::conj" "" source_text "${source_text}")
    string(REPLACE ".real()" "" source_text "${source_text}")
    file(WRITE "${QNPEPS_EO_GENERATED_DIR}/${source_name}.cpp" "${source_text}")
  endforeach()

  add_library(qnpeps_eo_oracle STATIC
    "${QNPEPS_EO_ORACLE_DIR}/boundary.cpp"
    "${QNPEPS_EO_ORACLE_DIR}/mps.cpp"
    "${QNPEPS_EO_ORACLE_DIR}/fixture.cpp"
    "${QNPEPS_EO_ORACLE_DIR}/linalg.cpp"
    "${QNPEPS_EO_GENERATED_DIR}/eloc.cpp"
    "${QNPEPS_EO_GENERATED_DIR}/hamiltonian.cpp"
    "${QNPEPS_EO_GENERATED_DIR}/ok.cpp"
    "${QNPEPS_EO_GENERATED_DIR}/minsr.cpp"
  )
  target_include_directories(qnpeps_eo_oracle PUBLIC "${QNPEPS_EO_ORACLE_DIR}")
  target_link_libraries(qnpeps_eo_oracle PUBLIC BLAS::BLAS)

  add_executable(qnpeps_peps_init_gate test/peps_init_gate.cu)
  target_include_directories(qnpeps_peps_init_gate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(qnpeps_peps_init_gate PRIVATE qnpeps CUDA::cudart)
  add_test(NAME qnpeps_peps_init_gate COMMAND qnpeps_peps_init_gate)

  add_executable(qnpeps_gram_gate test/gram_gate.cu)
  target_include_directories(qnpeps_gram_gate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(qnpeps_gram_gate PRIVATE qnpeps CUDA::cudart)
  add_test(NAME qnpeps_gram_gate COMMAND qnpeps_gram_gate)

  add_executable(qnpeps_minsr_gate test/minsr_gate.cu)
  target_include_directories(qnpeps_minsr_gate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(qnpeps_minsr_gate PRIVATE qnpeps CUDA::cudart)
  add_test(NAME qnpeps_minsr_gate COMMAND qnpeps_minsr_gate)

  add_executable(qnpeps_eo_core_test test/eo/core_test.cpp)
  target_include_directories(qnpeps_eo_core_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/eo
  )
  target_link_libraries(qnpeps_eo_core_test PRIVATE CUDA::cudart)
  add_test(NAME qnpeps_eo_core_test COMMAND qnpeps_eo_core_test)

  add_executable(qnpeps_eo_smoke test/eo/smoke.cu)
  target_include_directories(qnpeps_eo_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/eo)
  target_link_libraries(qnpeps_eo_smoke PRIVATE qnpeps CUDA::cudart)
  add_test(NAME qnpeps_eo_smoke COMMAND qnpeps_eo_smoke)

  add_executable(qnpeps_eo_suite test/eo/eo_suite.cu)
  target_include_directories(qnpeps_eo_suite PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/eo
    ${QNPEPS_EO_ORACLE_DIR}
  )
  target_link_libraries(qnpeps_eo_suite PRIVATE qnpeps qnpeps_eo_oracle CUDA::cudart)
  add_test(NAME qnpeps_eo_suite COMMAND qnpeps_eo_suite)

  foreach(gate_name gate_step1 gate_step2 gate_step3)
    add_executable(qnpeps_eo_${gate_name} test/eo/${gate_name}.cu)
    target_include_directories(qnpeps_eo_${gate_name} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/eo
      ${QNPEPS_EO_ORACLE_DIR}
      ${QNPEPS_EO_ORACLE_DIR}/cuda
    )
    target_link_libraries(qnpeps_eo_${gate_name} PRIVATE
      qnpeps qnpeps_eo_oracle CUDA::cudart
    )
    add_test(NAME qnpeps_eo_${gate_name} COMMAND qnpeps_eo_${gate_name})
  endforeach()

  add_executable(qnpeps_eo_l16_gate test/eo/gate_l16.cu)
  target_include_directories(qnpeps_eo_l16_gate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/eo)
  target_link_libraries(qnpeps_eo_l16_gate PRIVATE qnpeps CUDA::cudart)
  add_test(NAME qnpeps_eo_l16_gate COMMAND qnpeps_eo_l16_gate)

  foreach(gate_name gram minsr node step step_mg update)
    add_executable(qnpeps_e2e_${gate_name}_gate test/e2e/gate_${gate_name}.cu)
    target_include_directories(qnpeps_e2e_${gate_name}_gate PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/eo
      ${CMAKE_CURRENT_SOURCE_DIR}/e2e
      ${CMAKE_CURRENT_SOURCE_DIR}/test/e2e
      ${CMAKE_CURRENT_SOURCE_DIR}/test/eo
      ${QNPEPS_EO_ORACLE_DIR}
      ${QNPEPS_EO_ORACLE_DIR}/cuda
    )
    target_link_libraries(qnpeps_e2e_${gate_name}_gate PRIVATE
      qnpeps qnpeps_eo_oracle CUDA::cudart CUDA::cublas CUDA::cusolver Threads::Threads
    )
    add_test(NAME qnpeps_e2e_${gate_name}_gate COMMAND qnpeps_e2e_${gate_name}_gate)
  endforeach()
  target_sources(qnpeps_e2e_update_gate PRIVATE e2e/update.cu e2e/kernels.cu)
endif()
