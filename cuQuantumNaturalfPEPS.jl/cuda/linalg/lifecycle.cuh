#ifndef QNPEPS_LINALG_LIFECYCLE_CUH
#define QNPEPS_LINALG_LIFECYCLE_CUH

#include "core/types.cuh"
#include "linalg/handles.cuh"
#include "linalg/linalg.cuh"
#include "linalg/scratch.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace qnpeps
{
inline Linalg::~Linalg()
{
    destroy();
}

[[nodiscard]] inline auto Linalg::cublas() const noexcept -> cublasHandle_t
{
    return blas_ ? blas_->get() : nullptr;
}

[[nodiscard]] inline auto Linalg::cusolver() const noexcept -> cusolverDnHandle_t
{
    return solver_ ? solver_->get() : nullptr;
}

[[nodiscard]] inline auto Linalg::stream() const noexcept -> cudaStream_t
{
    return stream_;
}

[[nodiscard]] inline auto Linalg::device() const noexcept -> int
{
    return device_;
}

[[nodiscard]] inline auto Linalg::persistent_arena() noexcept -> ArenaCursor&
{
    return device_arena_->persistent_cursor();
}

[[nodiscard]] inline auto Linalg::transient_arena() -> TransientArenaCursor
{
    return device_arena_->transient_cursor(stream_);
}

[[nodiscard]] inline auto Linalg::arena_capacity() const noexcept -> usize
{
    return device_arena_->capacity();
}
}

#endif
