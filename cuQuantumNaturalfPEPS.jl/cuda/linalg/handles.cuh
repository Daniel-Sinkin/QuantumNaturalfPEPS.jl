#ifndef QNPEPS_LINALG_HANDLES_CUH
#define QNPEPS_LINALG_HANDLES_CUH

#include "core/cuda_utils.cuh"

#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <type_traits>
#include <utility>

namespace qnpeps
{
class CublasHandle
{
  public:
    ~CublasHandle()
    {
        if (handle_) unchecked(cublasDestroy(handle_));
    }

    CublasHandle(const CublasHandle&) = delete;
    auto operator=(const CublasHandle&) -> CublasHandle& = delete;

    CublasHandle(CublasHandle&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}

    auto operator=(CublasHandle&& other) noexcept -> CublasHandle&
    {
        if (this == &other) return *this;
        if (handle_) unchecked(cublasDestroy(handle_));
        handle_ = std::exchange(other.handle_, nullptr);
        return *this;
    }

    [[nodiscard]] static auto create(cudaStream_t stream) -> CublasHandle
    {
        cublasHandle_t handle{};
        check(cublasCreate(&handle));
        auto owner = CublasHandle{handle};
        check(cublasSetStream(owner.get(), stream));
        check(cublasSetMathMode(owner.get(), CUBLAS_DEFAULT_MATH));
        return owner;
    }

    [[nodiscard]] auto get() const noexcept -> cublasHandle_t
    {
        assert(handle_);
        return handle_;
    }

  private:
    explicit CublasHandle(cublasHandle_t handle) : handle_(handle) { assert(handle_); }

    cublasHandle_t handle_;
};

class CusolverDnHandle
{
  public:
    ~CusolverDnHandle()
    {
        if (handle_) unchecked(cusolverDnDestroy(handle_));
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
        if (handle_) unchecked(cusolverDnDestroy(handle_));
        handle_ = std::exchange(other.handle_, nullptr);
        return *this;
    }

    [[nodiscard]] static auto create(cudaStream_t stream) -> CusolverDnHandle
    {
        cusolverDnHandle_t handle{};
        check(cusolverDnCreate(&handle));
        auto owner = CusolverDnHandle{handle};
        check(cusolverDnSetStream(owner.get(), stream));
        return owner;
    }

    [[nodiscard]] auto get() const noexcept -> cusolverDnHandle_t
    {
        assert(handle_);
        return handle_;
    }

  private:
    explicit CusolverDnHandle(cusolverDnHandle_t handle) : handle_(handle) { assert(handle_); }

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
