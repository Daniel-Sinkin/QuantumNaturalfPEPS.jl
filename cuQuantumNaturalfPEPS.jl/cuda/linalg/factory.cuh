#ifndef QNPEPS_LINALG_FACTORY_CUH
#define QNPEPS_LINALG_FACTORY_CUH

#include "core/arena_cursor.cuh"
#include "core/cuda_utils.cuh"
#include "core/types.cuh"
#include "linalg/handles.cuh"
#include "linalg/linalg.cuh"

#include <cuda_runtime.h>
#include <memory>
#include <new>
#include <utility>

namespace qnpeps
{
[[nodiscard]] inline auto make_linalg(cudaStream_t stream) -> std::unique_ptr<Linalg>
{
    if (err_state() != QNPEPS_OK) return nullptr;
    if (not stream)
    {
        set_err(QNPEPS_ERR_BAD_CONFIG);
        return nullptr;
    }
    auto device = int{};
    CUDA_CHECK(cudaGetDevice(&device));
    if (err_state() != QNPEPS_OK) return nullptr;
    auto blas = CublasHandle::create(stream);
    if (not blas) return nullptr;
    auto solver = CusolverDnHandle::create(stream);
    if (not solver) return nullptr;
    auto device_arena = shared_device_arena(device);
    if (not device_arena) return nullptr;
    auto linalg = std::unique_ptr<Linalg>{new (std::nothrow) Linalg{
        device, stream, std::move(*blas), std::move(*solver), std::move(device_arena)
    }};
    if (not linalg)
    {
        set_err(QNPEPS_ERR_OOM);
        return nullptr;
    }
    return linalg;
}

struct RangefinderArgs
{
    CuMatrixBatchedCF32Const input{};
    int rank{};
    const cuFloatComplex* omega{};
    CuMatrixBatchedCF32 q_out{};
    CuMatrixBatchedCF32 r_out{};
    int dim_batch{};
    CuSpanCF32 sketch{};
    CuSpanCF32 projection{};
    CuSpanCF32 gram{};
    cuFloatComplex** gram_ptrs{};
    cuFloatComplex** sketch_ptrs{};
    int* info{};
    int* fail_flag{};
};

auto batched_rangefinder(Linalg& linalg, const RangefinderArgs& args) -> void;
}

#endif
