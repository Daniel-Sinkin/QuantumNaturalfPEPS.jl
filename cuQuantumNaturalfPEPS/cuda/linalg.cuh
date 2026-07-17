#ifndef QNPEPS_LINALG_CUH
#define QNPEPS_LINALG_CUH

#include "cuda_utils.cuh"
#include "types.cuh"

#include <algorithm>
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <optional>

namespace qnpeps
{
enum class BlasOp
{
    none,
    trans,
    conj_trans
};

[[nodiscard]] inline constexpr auto to_cublas(BlasOp op) -> cublasOperation_t
{
    // clang-format off
    switch (op) {
        case BlasOp::none      : return CUBLAS_OP_N;
        case BlasOp::trans     : return CUBLAS_OP_T;
        case BlasOp::conj_trans: return CUBLAS_OP_C;
    }
    // clang-format on
    __builtin_unreachable();
}

[[nodiscard]] inline constexpr auto from_cublas(cublasOperation_t op) -> BlasOp
{
    // clang-format off
    switch (op) {
        case CUBLAS_OP_N    : return BlasOp::none;
        case CUBLAS_OP_T    : return BlasOp::trans;
        case CUBLAS_OP_C    : return BlasOp::conj_trans;
        case CUBLAS_OP_CONJG: break;
    }
    // clang-format on
    __builtin_unreachable();
}

enum class BlasFillMode
{
    lower,
    upper,
    full
};

[[nodiscard]] inline constexpr auto to_cublas(BlasFillMode mode) -> cublasFillMode_t
{
    // clang-format off
    switch (mode) {
        case BlasFillMode::lower: return CUBLAS_FILL_MODE_LOWER;
        case BlasFillMode::upper: return CUBLAS_FILL_MODE_UPPER;
        case BlasFillMode::full : return CUBLAS_FILL_MODE_FULL;
    }
    // clang-format on
    __builtin_unreachable();
}

[[nodiscard]] inline constexpr auto from_cublas(cublasFillMode_t mode) -> BlasFillMode
{
    // clang-format off
    switch (mode) {
        case CUBLAS_FILL_MODE_LOWER: return BlasFillMode::lower;
        case CUBLAS_FILL_MODE_UPPER: return BlasFillMode::upper;
        case CUBLAS_FILL_MODE_FULL : return BlasFillMode::full;
    }
    // clang-format on
    __builtin_unreachable();
}

[[nodiscard]] inline constexpr auto op_rows(BlasOp op, int rows, int cols) noexcept -> int
{
    return op == BlasOp::trans or op == BlasOp::conj_trans ? cols : rows;
}
[[nodiscard]] inline constexpr auto op_cols(BlasOp op, int rows, int cols) noexcept -> int
{
    return op == BlasOp::trans or op == BlasOp::conj_trans ? rows : cols;
}

struct MatmulConfig
{
    BlasOp op_a{BlasOp::none};
    BlasOp op_b{BlasOp::none};
    f32 alpha_real{1.0f};
    f32 alpha_imag{0.0f};
    f32 beta_real{0.0f};
    f32 beta_imag{0.0f};
};

struct TriangularSolveConfig
{
    bool side_right{true};
    BlasFillMode fill_mode{BlasFillMode::lower};
    bool has_diag{true};
    f32 alpha_real{1.0f};
    f32 alpha_imag{0.0f};
    BlasOp op{BlasOp::none};
};

struct MatmulShape
{
    int m{};
    int k{};
    int n{};
};

[[nodiscard]] inline constexpr auto matmul_shape(
    const MatmulConfig& cfg, int a_rows, int a_cols, int b_rows, int b_cols, int c_rows, int c_cols
) noexcept -> std::optional<MatmulShape>
{
    if (a_rows <= 0 or a_cols <= 0 or b_rows <= 0 or b_cols <= 0 or c_rows <= 0 or c_cols <= 0)
        return std::nullopt;
    const auto m = op_rows(cfg.op_a, a_rows, a_cols);
    const auto k = op_cols(cfg.op_a, a_rows, a_cols);
    const auto n = op_cols(cfg.op_b, b_rows, b_cols);
    if (op_rows(cfg.op_b, b_rows, b_cols) != k or c_rows != m or c_cols != n) return std::nullopt;
    return MatmulShape{m, k, n};
}

class Linalg
{
  public:
    auto create(cudaStream_t s) -> void
    {
        if (h_ or sol_)
        {
            qnpeps::set_err(QNPEPS_ERR_INTERNAL);
            return;
        }
        assert(not h_ and not sol_);
        CUBLAS_CHECK(cublasCreate(&h_));
        CUSOLVER_CHECK(cusolverDnCreate(&sol_));
        stream_ = s;
        CUBLAS_CHECK(cublasSetStream(h_, s));
        CUSOLVER_CHECK(cusolverDnSetStream(sol_, s));
        CUBLAS_CHECK(cublasSetMathMode(h_, CUBLAS_DEFAULT_MATH));
    }
    auto destroy() -> void
    {
        if (h_) CUDA_NOCHECK(cublasDestroy(h_));
        if (sol_) CUDA_NOCHECK(cusolverDnDestroy(sol_));
        h_ = nullptr;
        sol_ = nullptr;
    }

    [[nodiscard]] auto blas() const -> cublasHandle_t { return h_; }
    [[nodiscard]] auto cusolver() const -> cusolverDnHandle_t { return sol_; }
    [[nodiscard]] auto stream() const -> cudaStream_t { return stream_; }

    auto
    matmul(CuMatrixConst A, CuMatrixConst B, CuMatrix C, const MatmulConfig& cfg = MatmulConfig{})
        -> void
    {
        const auto shape =
            matmul_shape(cfg, A.rows(), A.cols(), B.rows(), B.cols(), C.rows(), C.cols());
        const bool valid{shape.has_value() and A.data() and B.data() and C.data() and h_};
        if (not valid)
        {
            qnpeps::set_err(QNPEPS_ERR_INTERNAL);
            return;
        }
        assert(valid);
        const cuFloatComplex alpha{cfg.alpha_real, cfg.alpha_imag};
        const cuFloatComplex beta{cfg.beta_real, cfg.beta_imag};
        CUBLAS_CHECK(cublasCgemm(
            h_,
            to_cublas(cfg.op_a),
            to_cublas(cfg.op_b),
            shape->m,
            shape->n,
            shape->k,
            &alpha,
            cu_cast(A.data()),
            A.ld(),
            cu_cast(B.data()),
            B.ld(),
            &beta,
            cu_cast(C.data()),
            C.ld()
        ));
    }

    auto matmul_left_adj(CuMatrixConst A, CuMatrixConst B, CuMatrix C) -> void
    {
        matmul(A, B, C, {.op_a = BlasOp::conj_trans, .op_b = BlasOp::none});
    }

    auto matmul_batched(
        CuMatrixConstBatched A,
        CuMatrixConstBatched B,
        CuMatrixBatched C,
        int batch_size,
        const MatmulConfig& cfg
    ) -> void
    {
        const auto shape =
            matmul_shape(cfg, A.rows(), A.cols(), B.rows(), B.cols(), C.rows(), C.cols());
        const auto stored_elements = [](int rows, int cols) -> i64
        { return static_cast<i64>(rows) * cols; };
        const bool valid{
            shape.has_value() and batch_size > 0 and A.data() and B.data() and C.data() and h_
            and (A.stride() == 0 or A.stride() >= stored_elements(A.rows(), A.cols()))
            and (B.stride() == 0 or B.stride() >= stored_elements(B.rows(), B.cols()))
            and C.stride() >= stored_elements(C.rows(), C.cols())
        };
        if (not valid)
        {
            qnpeps::set_err(QNPEPS_ERR_INTERNAL);
            return;
        }
        assert(valid);
        const cuFloatComplex alpha{cfg.alpha_real, cfg.alpha_imag};
        const cuFloatComplex beta{cfg.beta_real, cfg.beta_imag};
        CUBLAS_CHECK(cublasCgemmStridedBatched(
            h_,
            to_cublas(cfg.op_a),
            to_cublas(cfg.op_b),
            shape->m,
            shape->n,
            shape->k,
            &alpha,
            cu_cast(A.data()),
            A.ld(),
            A.stride(),
            cu_cast(B.data()),
            B.ld(),
            B.stride(),
            &beta,
            cu_cast(C.data()),
            C.ld(),
            C.stride(),
            batch_size
        ));
    }

    auto matmul_batched(
        CuMatrixConstBatched A, CuMatrixConstBatched B, CuMatrixBatched C, int batch_size
    ) -> void
    {
        matmul_batched(A, B, C, batch_size, MatmulConfig{});
    }

    auto matmul_batched_left_adj(
        CuMatrixConstBatched A, CuMatrixConstBatched B, CuMatrixBatched C, int batch_size
    ) -> void
    {
        matmul_batched(A, B, C, batch_size, {.op_a = BlasOp::conj_trans, .op_b = BlasOp::none});
    }

    auto matmul_batched_right_adj(
        CuMatrixConstBatched A, CuMatrixConstBatched B, CuMatrixBatched C, int batch_size
    ) -> void
    {
        matmul_batched(A, B, C, batch_size, {.op_a = BlasOp::none, .op_b = BlasOp::conj_trans});
    }

    auto matmul_batched_both_adj(
        CuMatrixConstBatched A, CuMatrixConstBatched B, CuMatrixBatched C, int batch_size
    ) -> void
    {
        matmul_batched(
            A, B, C, batch_size, {.op_a = BlasOp::conj_trans, .op_b = BlasOp::conj_trans}
        );
    }

    auto matmul_batched_ptr(
        cf* const* a_array,
        int a_rows,
        int a_cols,
        cf* const* b_array,
        int b_rows,
        int b_cols,
        cf* const* c_array,
        int c_rows,
        int c_cols,
        int batch_size,
        const MatmulConfig& cfg = MatmulConfig{}
    ) -> void
    {
        const auto shape = matmul_shape(cfg, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols);
        const bool valid{
            shape.has_value() and batch_size > 0 and a_array and b_array and c_array and h_
        };
        if (not valid)
        {
            qnpeps::set_err(QNPEPS_ERR_INTERNAL);
            return;
        }
        assert(valid);
        const cuFloatComplex alpha{cfg.alpha_real, cfg.alpha_imag};
        const cuFloatComplex beta{cfg.beta_real, cfg.beta_imag};
        CUBLAS_CHECK(cublasCgemmBatched(
            h_,
            to_cublas(cfg.op_a),
            to_cublas(cfg.op_b),
            shape->m,
            shape->n,
            shape->k,
            &alpha,
            cu_cast(a_array),
            a_rows,
            cu_cast(b_array),
            b_rows,
            &beta,
            cu_cast(c_array),
            c_rows,
            batch_size
        ));
    }

    auto cholesky_lower_batched(int n, cf** as, int lda, int* info, int batch_size) -> void
    {
        const bool valid{n > 0 and as and lda >= n and info and batch_size > 0 and sol_};
        if (not valid)
        {
            qnpeps::set_err(QNPEPS_ERR_INTERNAL);
            return;
        }
        assert(valid);
        CUSOLVER_CHECK(cusolverDnCpotrfBatched(
            sol_, CUBLAS_FILL_MODE_LOWER, n, cu_cast(as), lda, info, batch_size
        ));
    }

    auto triangular_solve_batched(
        cf* const* as,
        int lda,
        cf* const* bs,
        int ldb,
        int m,
        int n,
        int batch_size,
        const TriangularSolveConfig& cfg = TriangularSolveConfig{}
    ) -> void
    {
        const auto triangular_dim = cfg.side_right ? n : m;
        const bool valid{
            as and bs and m > 0 and n > 0 and lda >= triangular_dim and ldb >= m and batch_size > 0
            and h_
        };
        if (not valid)
        {
            qnpeps::set_err(QNPEPS_ERR_INTERNAL);
            return;
        }
        assert(valid);
        const cuFloatComplex alpha{cfg.alpha_real, cfg.alpha_imag};
        CUBLAS_CHECK(cublasCtrsmBatched(
            h_,
            cfg.side_right ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT,
            to_cublas(cfg.fill_mode),
            to_cublas(cfg.op),
            cfg.has_diag ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT,
            m,
            n,
            &alpha,
            cu_cast(as),
            lda,
            cu_cast(bs),
            ldb,
            batch_size
        ));
    }

  private:
    cublasHandle_t h_{};
    cusolverDnHandle_t sol_{};
    cudaStream_t stream_{};
};

struct RangefinderArgs
{
    CuMatrixConstBatched input{};
    int k{};
    const cf* omega{};
    CuMatrixBatched q_out{};
    CuMatrixBatched r_out{};
    int dim_batch{};
    CuArray sketch{};
    CuArray proj{};
    CuArray gram{};
    cf** gram_ptrs{};
    cf** sketch_ptrs{};
    int* info{};
    int* fail_flag{};
};

auto batched_rangefinder(Linalg& la, const RangefinderArgs& args) -> void;

struct QrScratch
{
    usize reflector_bytes{};
    usize status_bytes{};
    usize workspace_bytes{};

    [[nodiscard]] constexpr auto total() const noexcept -> usize
    {
        return reflector_bytes + status_bytes + workspace_bytes;
    }
};

[[nodiscard]] inline auto qr_scratch(Linalg& la, int rows, int cols) -> QrScratch
{
    const bool valid{rows > 0 and cols > 0 and rows >= cols and la.cusolver()};
    if (not valid)
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return {};
    }
    assert(valid);
    int geqrf_size{};
    CUSOLVER_CHECK(
        cusolverDnCgeqrf_bufferSize(la.cusolver(), rows, cols, nullptr, rows, &geqrf_size)
    );
    int ungqr_size{};
    CUSOLVER_CHECK(cusolverDnCungqr_bufferSize(
        la.cusolver(), rows, cols, cols, nullptr, rows, nullptr, &ungqr_size
    ));
    const auto num_cols = static_cast<usize>(cols);
    const auto workspace_count = static_cast<usize>(std::max({geqrf_size, ungqr_size, 1}));
    return QrScratch{
        .reflector_bytes = device_align(num_cols * sizeof(cuFloatComplex)),
        .status_bytes = device_align(sizeof(int)),
        .workspace_bytes = device_align(sizeof(cuFloatComplex) * workspace_count),
    };
}

[[nodiscard]] inline auto qr_scratch(Linalg& la, CuMatrix A) -> QrScratch
{
    return qr_scratch(la, A.rows(), A.cols());
}

inline auto qr(Linalg& la, int m, int n, cf* A, int lda, void* scratch, const QrScratch& layout)
    -> void
{
    const bool valid{
        m > 0 and n > 0 and m >= n and A and lda >= m and scratch and layout.total() > 0
        and la.cusolver()
    };
    if (not valid)
    {
        qnpeps::set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
    assert(valid);
    auto* reflector_scalars = byte_offset<cuFloatComplex>(scratch, 0);
    auto* device_status = byte_offset<int>(scratch, layout.reflector_bytes);
    auto* solver_workspace =
        byte_offset<cuFloatComplex>(scratch, layout.reflector_bytes + layout.status_bytes);
    auto* a_cuda = cu_cast(A);
    const auto workspace_size = static_cast<int>(layout.workspace_bytes / sizeof(cuFloatComplex));
    CUSOLVER_CHECK(cusolverDnCgeqrf(
        la.cusolver(),
        m,
        n,
        a_cuda,
        lda,
        reflector_scalars,
        solver_workspace,
        workspace_size,
        device_status
    ));
    CUSOLVER_CHECK(cusolverDnCungqr(
        la.cusolver(),
        m,
        n,
        n,
        a_cuda,
        lda,
        reflector_scalars,
        solver_workspace,
        workspace_size,
        device_status
    ));
}

inline auto qr(Linalg& la, CuMatrix A, void* scratch, const QrScratch& layout) -> void
{
    qr(la, A.rows(), A.cols(), A.data(), A.ld(), scratch, layout);
}
}

#endif
