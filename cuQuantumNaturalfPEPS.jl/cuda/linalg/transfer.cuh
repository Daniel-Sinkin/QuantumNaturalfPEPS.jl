#ifndef QNPEPS_LINALG_TRANSFER_CUH
#define QNPEPS_LINALG_TRANSFER_CUH

#include "linalg/linalg.cuh"

namespace qnpeps
{
template <typename T>
auto upload_async(Linalg& linalg, T* destination, const T* source, usize count) noexcept -> void
{
    CUDA_CHECK(cudaMemcpyAsync(
        destination, source, count * sizeof(T), cudaMemcpyHostToDevice, linalg.stream()
    ));
}

template <typename T>
auto upload_async(Linalg& linalg, T** destination, const void* const* source, usize count) noexcept
    -> void
{
    static_assert(sizeof(T*) == sizeof(void*));
    CUDA_CHECK(cudaMemcpyAsync(
        destination, source, count * sizeof(T*), cudaMemcpyHostToDevice, linalg.stream()
    ));
}

template <typename T>
auto download_async(Linalg& linalg, T* destination, const T* source, usize count) noexcept -> void
{
    CUDA_CHECK(cudaMemcpyAsync(
        destination, source, count * sizeof(T), cudaMemcpyDeviceToHost, linalg.stream()
    ));
}

template <typename T>
auto copy_device_async(Linalg& linalg, T* destination, const T* source, usize count) noexcept
    -> void
{
    CUDA_CHECK(cudaMemcpyAsync(
        destination, source, count * sizeof(T), cudaMemcpyDeviceToDevice, linalg.stream()
    ));
}

template <typename T>
auto upload(T* destination, const T* source, usize count) noexcept -> void
{
    CUDA_CHECK(cudaMemcpy(destination, source, count * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
auto download(T* destination, const T* source, usize count) noexcept -> void
{
    CUDA_CHECK(cudaMemcpy(destination, source, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
auto copy_device(T* destination, const T* source, usize count) noexcept -> void
{
    CUDA_CHECK(cudaMemcpy(destination, source, count * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
auto copy_device_2d_async(
    Linalg& linalg,
    T* destination,
    usize destination_pitch,
    const T* source,
    usize source_pitch,
    usize width,
    usize height
) noexcept -> void
{
    CUDA_CHECK(cudaMemcpy2DAsync(
        destination,
        destination_pitch * sizeof(T),
        source,
        source_pitch * sizeof(T),
        width * sizeof(T),
        height,
        cudaMemcpyDeviceToDevice,
        linalg.stream()
    ));
}

template <typename T>
auto copy_peer_async(
    Linalg& linalg,
    T* destination,
    int destination_device,
    const T* source,
    int source_device,
    usize count
) noexcept -> void
{
    CUDA_CHECK(cudaMemcpyPeerAsync(
        destination, destination_device, source, source_device, count * sizeof(T), linalg.stream()
    ));
}

template <typename T>
auto zero_async(Linalg& linalg, T* destination, usize count) noexcept -> void
{
    CUDA_CHECK(cudaMemsetAsync(destination, 0, count * sizeof(T), linalg.stream()));
}
}

#endif
