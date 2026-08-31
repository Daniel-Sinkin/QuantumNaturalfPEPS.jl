#ifndef QNPEPS_SESSION_CUH
#define QNPEPS_SESSION_CUH

#include "core/cuda_utils.cuh"
#include "linalg/linalg.cuh"

#include <cuda_runtime.h>
#include <memory>
#include <new>
#include <utility>

namespace qnpeps
{
class Session
{
  public:
    ~Session()
    {
        int caller_device{};
        const auto have_caller = cudaGetDevice(&caller_device) == cudaSuccess;
        const auto switched = have_caller and device_ >= 0 and caller_device != device_
                              and cudaSetDevice(device_) == cudaSuccess;
        linalg_.reset();
        if (owns_stream_ and stream_) CUDA_NOCHECK(cudaStreamDestroy(stream_));
        if (switched) CUDA_NOCHECK(cudaSetDevice(caller_device));
    }

    Session(const Session&) = delete;
    Session(Session&&) = delete;
    auto operator=(const Session&) -> Session& = delete;
    auto operator=(Session&&) -> Session& = delete;

    [[nodiscard]] auto device() const noexcept -> int { return device_; }
    [[nodiscard]] auto stream() const noexcept -> cudaStream_t { return stream_; }
    [[nodiscard]] auto linalg() noexcept -> Linalg& { return *linalg_; }

  private:
    friend auto make_session(cudaStream_t stream, unsigned stream_flags)
        -> std::unique_ptr<Session>;

    Session(int device, cudaStream_t stream, bool owns_stream, std::unique_ptr<Linalg> linalg)
        : device_(device), stream_(stream), owns_stream_(owns_stream), linalg_(std::move(linalg))
    {
    }

    int device_{-1};
    cudaStream_t stream_{};
    bool owns_stream_{};
    std::unique_ptr<Linalg> linalg_{};
};

[[nodiscard]] inline auto make_session(
    cudaStream_t stream, unsigned stream_flags = cudaStreamDefault
) -> std::unique_ptr<Session>
{
    if (err_state() != QNPEPS_OK) return nullptr;
    int device{};
    CUDA_CHECK(cudaGetDevice(&device));
    if (err_state() != QNPEPS_OK) return nullptr;

    auto stream_use = stream;
    auto owns_stream = false;
    if (not stream_use)
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_use, stream_flags));
        if (err_state() != QNPEPS_OK) return nullptr;
        owns_stream = true;
    }

    auto linalg = make_linalg(stream_use);
    if (not linalg)
    {
        if (owns_stream) CUDA_NOCHECK(cudaStreamDestroy(stream_use));
        return nullptr;
    }
    std::unique_ptr<Session> session{
        new (std::nothrow) Session{device, stream_use, owns_stream, std::move(linalg)}
    };
    if (not session)
    {
        linalg.reset();
        if (owns_stream) CUDA_NOCHECK(cudaStreamDestroy(stream_use));
        set_err(QNPEPS_ERR_OOM);
    }
    return session;
}
}

#endif
