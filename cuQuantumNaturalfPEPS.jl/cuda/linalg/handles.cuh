#ifndef QNPEPS_LINALG_HANDLES_CUH
#define QNPEPS_LINALG_HANDLES_CUH

#include "core/cuda_utils.cuh"

#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <optional>
#include <type_traits>
#include <utility>

namespace qnpeps
{
class CublasHandle
{
  public:
    ~CublasHandle()
    {
        if (handle_) CUDA_NOCHECK(cublasDestroy(handle_));
    }

    CublasHandle(const CublasHandle&) = delete;
    auto operator=(const CublasHandle&) -> CublasHandle& = delete;

    CublasHandle(CublasHandle&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}

    auto operator=(CublasHandle&& other) noexcept -> CublasHandle&
    {
        if (this == &other) return *this;
        if (handle_) CUDA_NOCHECK(cublasDestroy(handle_));
        handle_ = std::exchange(other.handle_, nullptr);
        return *this;
    }

    [[nodiscard]] static auto create(cudaStream_t stream) -> std::optional<CublasHandle>
    {
        cublasHandle_t handle{};
        const auto create_status = cublasCreate(&handle);
        if (create_status != CUBLAS_STATUS_SUCCESS)
        {
            set_cublas_err(create_status);
            return std::nullopt;
        }

        auto owner = CublasHandle{handle};
        const auto stream_status = cublasSetStream(owner.get(), stream);
        if (stream_status != CUBLAS_STATUS_SUCCESS)
        {
            set_cublas_err(stream_status);
            return std::nullopt;
        }
        const auto math_status = cublasSetMathMode(owner.get(), CUBLAS_DEFAULT_MATH);
        if (math_status != CUBLAS_STATUS_SUCCESS)
        {
            set_cublas_err(math_status);
            return std::nullopt;
        }
        return std::optional<CublasHandle>{std::move(owner)};
    }

    [[nodiscard]] auto get() const noexcept -> cublasHandle_t
    {
        assert(handle_);
        return handle_;
    }

  private:
    explicit CublasHandle(cublasHandle_t handle) noexcept : handle_(handle) { assert(handle_); }

    cublasHandle_t handle_;
};

class CusolverDnHandle
{
  public:
    ~CusolverDnHandle()
    {
        if (handle_) CUDA_NOCHECK(cusolverDnDestroy(handle_));
    }

    CusolverDnHandle(const CusolverDnHandle&) = delete;
    auto operator=(const CusolverDnHandle&) -> CusolverDnHandle& = delete;

    CusolverDnHandle(CusolverDnHandle&& other) noexcept
        : handle_(std::exchange(other.handle_, nullptr))
    {
    }

    auto operator=(CusolverDnHandle&& other) noexcept -> CusolverDnHandle&
    {
        if (this == &other) return *this;
        if (handle_) CUDA_NOCHECK(cusolverDnDestroy(handle_));
        handle_ = std::exchange(other.handle_, nullptr);
        return *this;
    }

    [[nodiscard]] static auto create(cudaStream_t stream) -> std::optional<CusolverDnHandle>
    {
        cusolverDnHandle_t handle{};
        const auto create_status = cusolverDnCreate(&handle);
        if (create_status != CUSOLVER_STATUS_SUCCESS)
        {
            set_cusolver_err(create_status);
            return std::nullopt;
        }

        auto owner = CusolverDnHandle{handle};
        const auto stream_status = cusolverDnSetStream(owner.get(), stream);
        if (stream_status != CUSOLVER_STATUS_SUCCESS)
        {
            set_cusolver_err(stream_status);
            return std::nullopt;
        }
        return std::optional<CusolverDnHandle>{std::move(owner)};
    }

    [[nodiscard]] auto get() const noexcept -> cusolverDnHandle_t
    {
        assert(handle_);
        return handle_;
    }

  private:
    explicit CusolverDnHandle(cusolverDnHandle_t handle) noexcept : handle_(handle)
    {
        assert(handle_);
    }

    cusolverDnHandle_t handle_;
};

static_assert(not std::is_default_constructible_v<CublasHandle>);
static_assert(not std::is_copy_constructible_v<CublasHandle>);
static_assert(std::is_move_constructible_v<CublasHandle>);
static_assert(not std::is_default_constructible_v<CusolverDnHandle>);
static_assert(not std::is_copy_constructible_v<CusolverDnHandle>);
static_assert(std::is_move_constructible_v<CusolverDnHandle>);
}

#endif
