#include "eo.cuh"

namespace qnpeps
{

auto Linalg::destroy() -> void
{
    auto caller_device = int{};
    const auto have_caller = cudaGetDevice(&caller_device) == cudaSuccess;
    const auto switched = have_caller and device_ >= 0 and caller_device != device_
                          and cudaSetDevice(device_) == cudaSuccess;
    if (gesvdj_parameters_) CUDA_NOCHECK(cusolverDnDestroyGesvdjInfo(gesvdj_parameters_));
    gesvdj_parameters_ = nullptr;
    blas_.reset();
    solver_.reset();
    if (switched) CUDA_NOCHECK(cudaSetDevice(caller_device));
    device_ = -1;
    stream_ = nullptr;
}

auto Linalg::matmul(DeviceMatrix a, DeviceMatrix b, DeviceMatrix c) -> void
{
    if (a.cols() != b.rows() or c.rows() != a.rows() or c.cols() != b.cols())
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    const auto one = cuComplex{1.0f, 0.0f};
    const auto zero = cuComplex{0.0f, 0.0f};
    CUBLAS_CHECK(cublasCgemm(
        cublas(),
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        a.rows(),
        b.cols(),
        a.cols(),
        &one,
        reinterpret_cast<const cuComplex*>(a.data()),
        a.ld(),
        reinterpret_cast<const cuComplex*>(b.data()),
        b.ld(),
        &zero,
        reinterpret_cast<cuComplex*>(c.data()),
        c.ld()
    ));
}

auto Linalg::matmul_adj_norm(DeviceMatrix a, DeviceMatrix b, DeviceMatrix c) -> void
{
    if (a.rows() != b.rows() or c.rows() != a.cols() or c.cols() != b.cols())
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    const cuComplex one{1.0f, 0.0f};
    const cuComplex zero{0.0f, 0.0f};
    CUBLAS_CHECK(cublasCgemm(
        cublas(),
        CUBLAS_OP_C,
        CUBLAS_OP_N,
        a.cols(),
        b.cols(),
        a.rows(),
        &one,
        reinterpret_cast<const cuComplex*>(a.data()),
        a.ld(),
        reinterpret_cast<const cuComplex*>(b.data()),
        b.ld(),
        &zero,
        reinterpret_cast<cuComplex*>(c.data()),
        c.ld()
    ));
}

auto Linalg::cholesky_batched(int n, ComplexF32** arrays, int lda, int* info, int batch_size)
    -> void
{
    CUSOLVER_CHECK(cusolverDnCpotrfBatched(
        cusolver(),
        CUBLAS_FILL_MODE_LOWER,
        n,
        reinterpret_cast<cuComplex**>(arrays),
        lda,
        info,
        batch_size
    ));
}

auto Linalg::solve_triangular_batched(
    int m, int n, ComplexF32* const* a, int lda, ComplexF32* const* b, int ldb, int batch_size
) -> void
{
    const auto one = cuComplex{1.0f, 0.0f};
    CUBLAS_CHECK(cublasCtrsmBatched(
        cublas(),
        CUBLAS_SIDE_RIGHT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_C,
        CUBLAS_DIAG_NON_UNIT,
        m,
        n,
        &one,
        reinterpret_cast<cuComplex* const*>(a),
        lda,
        reinterpret_cast<cuComplex* const*>(b),
        ldb,
        batch_size
    ));
}

}
