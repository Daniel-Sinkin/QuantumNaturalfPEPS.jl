#ifndef QNPEPS_LINALG_FACTORY_CUH
#define QNPEPS_LINALG_FACTORY_CUH

#include "core/arena_cursor.cuh"
#include "core/cuda_utils.cuh"
#include "core/types.cuh"
#include "linalg/handles.cuh"
#include "linalg/linalg.cuh"

#include <cuda_runtime.h>
#include <memory>
#include <utility>

namespace qnpeps
{
[[nodiscard]] inline auto make_linalg(cudaStream_t stream) -> std::unique_ptr<Linalg>
{
    require(stream != nullptr, Status::bad_config, "linear algebra context needs a stream");
    auto device = int{};
    check(cudaGetDevice(&device));
    auto blas = CublasHandle::create(stream);
    auto solver = CusolverDnHandle::create(stream);
    auto device_arena = shared_device_arena(device);
    auto arena_lease = device_arena->acquire_lease(stream);
    return std::unique_ptr<Linalg>{new Linalg{
        device,
        stream,
        std::move(blas),
        std::move(solver),
        std::move(device_arena),
        std::move(arena_lease)
    }};
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
    bool svd{};
};

auto batched_rangefinder(Linalg& linalg, const RangefinderArgs& args) -> void;
}

#endif
