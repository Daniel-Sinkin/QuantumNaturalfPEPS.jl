#ifndef QNPEPS_SESSION_CUH
#define QNPEPS_SESSION_CUH

#include "core/cuda_utils.cuh"
#include "core/defer.cuh"
#include "linalg/linalg.cuh"

#include <cuda_runtime.h>
#include <memory>
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
        if (owns_stream_ and stream_) unchecked(cudaStreamDestroy(stream_));
        if (switched) unchecked(cudaSetDevice(caller_device));
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
    int device{};
    check(cudaGetDevice(&device));

    auto stream_use = stream;
    auto owns_stream = false;
    if (not stream_use)
    {
        check(cudaStreamCreateWithFlags(&stream_use, stream_flags));
        owns_stream = true;
    }
    auto stream_adopted = false;
    const Defer stream_cleanup{[&]
                               {
                                   if (owns_stream and not stream_adopted)
                                       unchecked(cudaStreamDestroy(stream_use));
                               }};

    auto linalg = make_linalg(stream_use);
    std::unique_ptr<Session> session{
        new Session{device, stream_use, owns_stream, std::move(linalg)}
    };
    stream_adopted = true;
    return session;
}
}

#endif
