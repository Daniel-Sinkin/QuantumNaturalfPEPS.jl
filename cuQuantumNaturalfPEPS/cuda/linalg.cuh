#ifndef QNPEPS_LINALG_CUH
#define QNPEPS_LINALG_CUH

#include "cuda_utils.cuh"
#include "types.cuh"

#include <algorithm>
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <memory>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

namespace qnpeps
{
enum class BlasOp
{
    none,
    trans,
    conj_trans
};
[[nodiscard]] inline constexpr auto is_trans(BlasOp op) noexcept -> bool
{
    return op == BlasOp::trans or op == BlasOp::conj_trans;
}

[[nodiscard]] inline constexpr auto to_cublas(BlasOp op) -> cublasOperation_t
{
    // clang-format off
    switch (op) {
        case BlasOp::none      : return CUBLAS_OP_N;
        case BlasOp::trans     : return CUBLAS_OP_T;
        case BlasOp::conj_trans: return CUBLAS_OP_C;
    }
    // clang-format on
    QNPEPS_UNREACHABLE();
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
    QNPEPS_UNREACHABLE();
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
    QNPEPS_UNREACHABLE();
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
    QNPEPS_UNREACHABLE();
}

[[nodiscard]] inline constexpr auto op_rows(BlasOp op, int rows, int cols) noexcept -> int
{
    return is_trans(op) ? cols : rows;
}
[[nodiscard]] inline constexpr auto op_cols(BlasOp op, int rows, int cols) noexcept -> int
{
    return is_trans(op) ? rows : cols;
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

class CublasHandle
{
  public:
    ~CublasHandle()
    {
        if (handle_) CUDA_NOCHECK(cublasDestroy(handle_));
    }

    CublasHandle(const CublasHandle&) = delete;
    auto operator=(const CublasHandle&) -> CublasHandle& = delete;

    CublasHandle(CublasHandle&& other) noexcept
        : handle_(std::exchange(other.handle_, nullptr))
    {
    }

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

        CublasHandle owner{handle};
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

        CusolverDnHandle owner{handle};
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

template <typename MatrixA, typename MatrixB, typename MatrixC>
[[nodiscard]] inline constexpr auto matmul_shape(
    const MatmulConfig& cfg, const MatrixA& a, const MatrixB& b, const MatrixC& c
) noexcept -> std::optional<MatmulShape>
{
    const auto valid_a = a.rows() > 0 and a.cols() > 0;
    const auto valid_b = b.rows() > 0 and b.cols() > 0;
    const auto valid_c = c.rows() > 0 and c.cols() > 0;
    if (not valid_a or not valid_b or not valid_c) return std::nullopt;

    const auto m = op_rows(cfg.op_a, a.rows(), a.cols());
    const auto k = op_cols(cfg.op_a, a.rows(), a.cols());
    const auto n = op_cols(cfg.op_b, b.rows(), b.cols());
    const auto compatible_b = op_rows(cfg.op_b, b.rows(), b.cols()) == k;
    const auto compatible_c = c.rows() == m and c.cols() == n;
    if (not compatible_b or not compatible_c) return std::nullopt;
    return MatmulShape{m, k, n};
}

class Linalg
{
  public:
    ~Linalg() = default;

    [[nodiscard]] auto stream() const -> cudaStream_t { return stream_; }

    [[nodiscard]] auto qr_scratch(int rows, int cols) -> QrScratch
    {
        if (rows <= 0 or cols <= 0 or rows < cols)
        {
            assert(false);
            qnpeps::set_err(QNPEPS_ERR_INTERNAL);
            return {};
        }
        int geqrf_size{};
        CUSOLVER_CHECK(
            cusolverDnCgeqrf_bufferSize(solver_.get(), rows, cols, nullptr, rows, &geqrf_size)
        );
        int ungqr_size{};
        CUSOLVER_CHECK(cusolverDnCungqr_bufferSize(
            solver_.get(), rows, cols, cols, nullptr, rows, nullptr, &ungqr_size
        ));
        const auto num_cols = static_cast<usize>(cols);
        const auto workspace_count = static_cast<usize>(std::max({geqrf_size, ungqr_size, 1}));
        return QrScratch{
            .reflector_bytes = device_align(num_cols * sizeof(cuFloatComplex)),
            .status_bytes = device_align(sizeof(int)),
            .workspace_bytes = device_align(sizeof(cuFloatComplex) * workspace_count),
        };
    }

    [[nodiscard]] auto qr_scratch(CuMatrix matrix) -> QrScratch
    {
        return qr_scratch(matrix.rows(), matrix.cols());
    }

    auto qr(CuMatrix matrix, void* scratch, const QrScratch& layout) -> void
    {
        const auto rows = matrix.rows();
        const auto cols = matrix.cols();
        const auto valid_dimensions = rows > 0 and cols > 0 and rows >= cols;
        const auto valid_matrix = matrix.data() != nullptr;
        const auto valid_scratch = scratch and layout.total() > 0;
        const auto valid = valid_dimensions and valid_matrix and valid_scratch;
        if (not valid)
        {
            assert(false);
            qnpeps::set_err(QNPEPS_ERR_INTERNAL);
            return;
        }
        auto* reflector_scalars = byte_offset<cuFloatComplex>(scratch, 0);
        auto* device_status = byte_offset<int>(scratch, layout.reflector_bytes);
        const auto workspace_offset = layout.reflector_bytes + layout.status_bytes;
        auto* solver_workspace = byte_offset<cuFloatComplex>(scratch, workspace_offset);
        const auto workspace_elements = layout.workspace_bytes / sizeof(cuFloatComplex);
        const auto workspace_size = static_cast<int>(workspace_elements);
        CUSOLVER_CHECK(cusolverDnCgeqrf(
            solver_.get(),
            rows,
            cols,
            matrix.data(),
            matrix.ld(),
            reflector_scalars,
            solver_workspace,
            workspace_size,
            device_status
        ));
        CUSOLVER_CHECK(cusolverDnCungqr(
            solver_.get(),
            rows,
            cols,
            cols,
            matrix.data(),
            matrix.ld(),
            reflector_scalars,
            solver_workspace,
            workspace_size,
            device_status
        ));
    }

    auto matmul(
        CuMatrixConst a, CuMatrixConst b, CuMatrix c, const MatmulConfig& cfg = MatmulConfig{}
    ) -> void
    {
        const auto shape = matmul_shape(cfg, a, b, c);
        const auto valid = shape.has_value() and a.data() and b.data() and c.data();
        if (not valid)
        {
            qnpeps::set_err(QNPEPS_ERR_INTERNAL);
            return;
        }
        const cuFloatComplex alpha{cfg.alpha_real, cfg.alpha_imag};
        const cuFloatComplex beta{cfg.beta_real, cfg.beta_imag};
        CUBLAS_CHECK(cublasCgemm(
            blas_.get(),
            to_cublas(cfg.op_a),
            to_cublas(cfg.op_b),
            shape->m,
            shape->n,
            shape->k,
            &alpha,
            a.data(),
            a.ld(),
            b.data(),
            b.ld(),
            &beta,
            c.data(),
            c.ld()
        ));
    }

    auto matmul_left_adj(CuMatrixConst a, CuMatrixConst b, CuMatrix c) -> void
    {
        matmul(a, b, c, {.op_a = BlasOp::conj_trans, .op_b = BlasOp::none});
    }

    auto matmul_batched(
        CuMatrixConstBatched a,
        CuMatrixConstBatched b,
        CuMatrixBatched c,
        int batch_size,
        const MatmulConfig& cfg
    ) -> void
    {
        const auto shape = matmul_shape(cfg, a, b, c);
        const auto stored_elements = [](CuMatrixConstBatched x) -> i64
        { return static_cast<i64>(x.rows()) * x.cols(); };
        const auto valid_shape = shape.has_value();
        const auto valid_batch = batch_size > 0;
        const auto valid_a = a.data() and (a.stride() == 0 or a.stride() >= stored_elements(a));
        const auto valid_b = b.data() and (b.stride() == 0 or b.stride() >= stored_elements(b));
        const auto valid_c = c.data() and c.stride() >= stored_elements(c);
        const auto valid = valid_shape and valid_batch and valid_a and valid_b and valid_c;
        if (not valid)
        {
            qnpeps::set_err(QNPEPS_ERR_INTERNAL);
            return;
        }
        const cuFloatComplex alpha{cfg.alpha_real, cfg.alpha_imag};
        const cuFloatComplex beta{cfg.beta_real, cfg.beta_imag};
        CUBLAS_CHECK(cublasCgemmStridedBatched(
            blas_.get(),
            to_cublas(cfg.op_a),
            to_cublas(cfg.op_b),
            shape->m,
            shape->n,
            shape->k,
            &alpha,
            a.data(),
            a.ld(),
            a.stride(),
            b.data(),
            b.ld(),
            b.stride(),
            &beta,
            c.data(),
            c.ld(),
            c.stride(),
            batch_size
        ));
    }

    auto matmul_batched(
        CuMatrixConstBatched a, CuMatrixConstBatched b, CuMatrixBatched c, int batch_size
    ) -> void
    {
        matmul_batched(a, b, c, batch_size, MatmulConfig{});
    }

    auto matmul_batched_left_adj(
        CuMatrixConstBatched a, CuMatrixConstBatched b, CuMatrixBatched c, int batch_size
    ) -> void
    {
        matmul_batched(a, b, c, batch_size, {.op_a = BlasOp::conj_trans, .op_b = BlasOp::none});
    }

    auto matmul_batched_right_adj(
        CuMatrixConstBatched a, CuMatrixConstBatched b, CuMatrixBatched c, int batch_size
    ) -> void
    {
        matmul_batched(a, b, c, batch_size, {.op_a = BlasOp::none, .op_b = BlasOp::conj_trans});
    }

    auto matmul_batched_both_adj(
        CuMatrixConstBatched a, CuMatrixConstBatched b, CuMatrixBatched c, int batch_size
    ) -> void
    {
        matmul_batched(
            a, b, c, batch_size, {.op_a = BlasOp::conj_trans, .op_b = BlasOp::conj_trans}
        );
    }

    auto matmul_batched_ptr(
        cuFloatComplex* const* a_array,
        int a_rows,
        int a_cols,
        cuFloatComplex* const* b_array,
        int b_rows,
        int b_cols,
        cuFloatComplex* const* c_array,
        int c_rows,
        int c_cols,
        int batch_size,
        const MatmulConfig& cfg = MatmulConfig{}
    ) -> void
    {
        const auto a = CuMatrixConst{nullptr, a_rows, a_cols};
        const auto b = CuMatrixConst{nullptr, b_rows, b_cols};
        const auto c = CuMatrix{nullptr, c_rows, c_cols};
        const auto shape = matmul_shape(cfg, a, b, c);
        const auto valid = shape.has_value() and batch_size > 0 and a_array and b_array and c_array;
        if (not valid)
        {
            qnpeps::set_err(QNPEPS_ERR_INTERNAL);
            return;
        }
        const cuFloatComplex alpha{cfg.alpha_real, cfg.alpha_imag};
        const cuFloatComplex beta{cfg.beta_real, cfg.beta_imag};
        CUBLAS_CHECK(cublasCgemmBatched(
            blas_.get(),
            to_cublas(cfg.op_a),
            to_cublas(cfg.op_b),
            shape->m,
            shape->n,
            shape->k,
            &alpha,
            a_array,
            a_rows,
            b_array,
            b_rows,
            &beta,
            c_array,
            c_rows,
            batch_size
        ));
    }

    auto cholesky_lower_batched(int n, cuFloatComplex** as, int lda, int* info, int batch_size)
        -> void
    {
        const auto valid = n > 0 and as and lda >= n and info and batch_size > 0;
        if (not valid)
        {
            qnpeps::set_err(QNPEPS_ERR_INTERNAL);
            return;
        }
        CUSOLVER_CHECK(
            cusolverDnCpotrfBatched(
                solver_.get(), CUBLAS_FILL_MODE_LOWER, n, as, lda, info, batch_size
            )
        );
    }

    auto triangular_solve_batched(
        cuFloatComplex* const* as,
        int lda,
        cuFloatComplex* const* bs,
        int ldb,
        int m,
        int n,
        int batch_size,
        const TriangularSolveConfig& cfg = TriangularSolveConfig{}
    ) -> void
    {
        const auto triangular_dim = cfg.side_right ? n : m;
        const auto pointers_valid = as and bs;
        const auto dimensions_valid = m > 0 and n > 0;
        const auto leading_dimensions_valid = lda >= triangular_dim and ldb >= m;
        const auto batch_valid = batch_size > 0;
        const auto shape_valid = dimensions_valid and leading_dimensions_valid;
        const auto valid = pointers_valid and shape_valid and batch_valid;
        if (not valid)
        {
            qnpeps::set_err(QNPEPS_ERR_INTERNAL);
            return;
        }
        const cuFloatComplex alpha{cfg.alpha_real, cfg.alpha_imag};
        CUBLAS_CHECK(cublasCtrsmBatched(
            blas_.get(),
            cfg.side_right ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT,
            to_cublas(cfg.fill_mode),
            to_cublas(cfg.op),
            cfg.has_diag ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT,
            m,
            n,
            &alpha,
            as,
            lda,
            bs,
            ldb,
            batch_size
        ));
    }

  private:
    friend auto make_linalg(cudaStream_t stream) -> std::unique_ptr<Linalg>;

    Linalg(const Linalg&) = delete;
    Linalg(Linalg&&) = delete;
    auto operator=(const Linalg&) -> Linalg& = delete;
    auto operator=(Linalg&&) -> Linalg& = delete;

    Linalg(CublasHandle&& blas, CusolverDnHandle&& solver, cudaStream_t stream) noexcept
        : blas_(std::move(blas)), solver_(std::move(solver)), stream_(stream)
    {
        assert(blas_.get());
        assert(solver_.get());
    }

    CublasHandle blas_;
    CusolverDnHandle solver_;
    cudaStream_t stream_{};
};

static_assert(not std::is_default_constructible_v<Linalg>);
static_assert(not std::is_copy_constructible_v<Linalg>);
static_assert(not std::is_move_constructible_v<Linalg>);

[[nodiscard]] inline auto make_linalg(cudaStream_t stream) -> std::unique_ptr<Linalg>
{
    if (err_state() != QNPEPS_OK) return nullptr;

    auto blas = CublasHandle::create(stream);
    if (not blas) return nullptr;
    auto solver = CusolverDnHandle::create(stream);
    if (not solver) return nullptr;

    auto linalg = std::unique_ptr<Linalg>{
        new (std::nothrow) Linalg{std::move(*blas), std::move(*solver), stream}
    };
    if (not linalg) set_err(QNPEPS_ERR_OOM);
    return linalg;
}

struct RangefinderArgs
{
    CuMatrixConstBatched input{};
    int rank{};
    const cuFloatComplex* omega{};
    CuMatrixBatched q_out{};
    CuMatrixBatched r_out{};
    int dim_batch{};
    CuArray sketch{};
    CuArray projection{};
    CuArray gram{};
    cuFloatComplex** gram_ptrs{};
    cuFloatComplex** sketch_ptrs{};
    int* info{};
    int* fail_flag{};
};

auto batched_rangefinder(Linalg& la, const RangefinderArgs& args) -> void;

}

#endif
