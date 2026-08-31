#ifndef QNPEPS_LINALG_CUH
#define QNPEPS_LINALG_CUH

#include "core/arena_cursor.cuh"
#include "core/cuda_utils.cuh"
#include "core/predicates.cuh"
#include "core/types.cuh"
#include "linalg/handles.cuh"
#include "linalg/scratch.cuh"

#include <algorithm>
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace qnpeps
{
class DeviceMatrix;
class MatrixBatch;

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

    switch (op)
    {
        case BlasOp::none:
            return CUBLAS_OP_N;
        case BlasOp::trans:
            return CUBLAS_OP_T;
        case BlasOp::conj_trans:
            return CUBLAS_OP_C;
    }

    QNPEPS_UNREACHABLE();
}

[[nodiscard]] inline constexpr auto from_cublas(cublasOperation_t op) -> BlasOp
{

    switch (op)
    {
        case CUBLAS_OP_N:
            return BlasOp::none;
        case CUBLAS_OP_T:
            return BlasOp::trans;
        case CUBLAS_OP_C:
            return BlasOp::conj_trans;
        case CUBLAS_OP_CONJG:
            break;
    }

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

    switch (mode)
    {
        case BlasFillMode::lower:
            return CUBLAS_FILL_MODE_LOWER;
        case BlasFillMode::upper:
            return CUBLAS_FILL_MODE_UPPER;
        case BlasFillMode::full:
            return CUBLAS_FILL_MODE_FULL;
    }

    QNPEPS_UNREACHABLE();
}

[[nodiscard]] inline constexpr auto from_cublas(cublasFillMode_t mode) -> BlasFillMode
{

    switch (mode)
    {
        case CUBLAS_FILL_MODE_LOWER:
            return BlasFillMode::lower;
        case CUBLAS_FILL_MODE_UPPER:
            return BlasFillMode::upper;
        case CUBLAS_FILL_MODE_FULL:
            return BlasFillMode::full;
    }

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

struct GemvConfig
{
    BlasOp op{BlasOp::none};
    f64 alpha_real{1.0};
    f64 alpha_imag{0.0};
    f64 beta_real{0.0};
    f64 beta_imag{0.0};
};

struct LinalgState
{
    cublasMath_t math{};
    cublasPointerMode_t pointer{};
    cublasAtomicsMode_t atomics{};
    cusolverDeterministicMode_t deterministic{};
};

struct BlasState
{
    cublasMath_t math{};
    cublasPointerMode_t pointer{};
    cublasAtomicsMode_t atomics{};
};

struct DiagonalizeConfig
{
    f64* eigenvalues;
    cuDoubleComplex* workspace;
    int workspace_count;
    int* info;
};

struct QrStageConfig
{
    cuFloatComplex* reflector_scalars;
    cuFloatComplex* workspace;
    int workspace_count;
    int* info;
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
    [[nodiscard]] auto diagonalize_workspace_count(int order) -> int;
    [[nodiscard]] auto diagonalize_workspace_count(CuMatrixCF64 matrix, f64* eigenvalues) -> int;
    auto diagonalize(CuMatrixCF64 matrix, const DiagonalizeConfig& config) -> void;
    [[nodiscard]] auto heevd_scratch(int order) -> HeevdScratch;
    auto zheevd(CuMatrixCF64 matrix, void* scratch, const HeevdScratch& layout) -> void;
    auto matmul(DeviceMatrix a, DeviceMatrix b, DeviceMatrix c) -> void;
    auto matmul_adj_norm(DeviceMatrix a, DeviceMatrix b, DeviceMatrix c) -> void;
    auto cholesky_batched(int n, ComplexF32** arrays, int lda, int* info, int batch_size) -> void;
    auto solve_triangular_batched(
        int m, int n, ComplexF32* const* a, int lda, ComplexF32* const* b, int ldb, int batch_size
    ) -> void;
    auto cholesky_lower_batched(int n, cuFloatComplex** as, int lda, int* info, int batch_size)
        -> void;
    auto triangular_solve_batched(
        cuFloatComplex* const* as,
        int lda,
        cuFloatComplex* const* bs,
        int ldb,
        int m,
        int n,
        int batch_size,
        const TriangularSolveConfig& cfg = TriangularSolveConfig{}
    ) -> void;
    [[nodiscard]] auto blas_state(BlasState& state) noexcept -> cublasStatus_t;
    [[nodiscard]] auto set_blas_state(const BlasState& state) noexcept -> cublasStatus_t;
    [[nodiscard]] auto set_gram_state() noexcept -> cublasStatus_t;
    [[nodiscard]] auto accumulate_hermitian(
        int rows,
        int dense_width,
        const cuFloatComplex* dense,
        cuFloatComplex* output,
        int output_ld
    ) noexcept -> cublasStatus_t;
    [[nodiscard]] auto accumulate_gram_block(
        int rows_a,
        int rows_b,
        int dense_width,
        const cuFloatComplex* dense_a,
        const cuFloatComplex* dense_b,
        cuFloatComplex* output,
        int output_ld
    ) noexcept -> cublasStatus_t;
    ~Linalg();

    [[nodiscard]] auto cublas() const noexcept -> cublasHandle_t;
    [[nodiscard]] auto cusolver() const noexcept -> cusolverDnHandle_t;
    [[nodiscard]] auto stream() const noexcept -> cudaStream_t;
    [[nodiscard]] auto device() const noexcept -> int;
    [[nodiscard]] auto persistent_arena() noexcept -> ArenaCursor&;
    [[nodiscard]] auto transient_arena() -> TransientArenaCursor;
    [[nodiscard]] auto arena_capacity() const noexcept -> usize;
    auto gemv(
        CuMatrixCF64Const a,
        const cuDoubleComplex* x,
        cuDoubleComplex* y,
        const GemvConfig& cfg = GemvConfig{}
    ) -> void;
    auto matmul(
        CuMatrixCF32Const a,
        CuMatrixCF32Const b,
        CuMatrixCF32 c,
        const MatmulConfig& cfg = MatmulConfig{}
    ) -> void;
    auto matmul(
        CuMatrixCF64Const a,
        CuMatrixCF64Const b,
        CuMatrixCF64 c,
        const MatmulConfig& cfg = MatmulConfig{}
    ) -> void;
    auto matmul_left_adj(CuMatrixCF32Const a, CuMatrixCF32Const b, CuMatrixCF32 c) -> void;
    auto matmul_batched(
        CuMatrixBatchedCF32Const a,
        CuMatrixBatchedCF32Const b,
        CuMatrixBatchedCF32 c,
        int batch_size,
        const MatmulConfig& cfg
    ) -> void;
    auto matmul_batched(
        CuMatrixBatchedCF32Const a,
        CuMatrixBatchedCF32Const b,
        CuMatrixBatchedCF32 c,
        int batch_size
    ) -> void;
    auto matmul_batched_left_adj(
        CuMatrixBatchedCF32Const a,
        CuMatrixBatchedCF32Const b,
        CuMatrixBatchedCF32 c,
        int batch_size
    ) -> void;
    auto matmul_batched_right_adj(
        CuMatrixBatchedCF32Const a,
        CuMatrixBatchedCF32Const b,
        CuMatrixBatchedCF32 c,
        int batch_size
    ) -> void;
    auto matmul_batched_both_adj(
        CuMatrixBatchedCF32Const a,
        CuMatrixBatchedCF32Const b,
        CuMatrixBatchedCF32 c,
        int batch_size
    ) -> void;
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
    ) -> void;
    [[nodiscard]] auto qr_workspace_count(int rows, int cols) -> int;
    [[nodiscard]] auto qr_scratch(int rows, int cols) -> QrScratch;
    [[nodiscard]] auto qr_scratch(CuMatrixCF32 matrix) -> QrScratch;
    auto qr(CuMatrixCF32 matrix, void* scratch, const QrScratch& layout) -> void;
    auto qr_factor(CuMatrixCF32 matrix, const QrStageConfig& config) -> void;
    auto qr_form(CuMatrixCF32 matrix, const QrStageConfig& config) -> void;
    [[nodiscard]] auto enter_default_state(LinalgState& state) noexcept -> bool;
    auto restore_state(const LinalgState& state) noexcept -> void;
    [[nodiscard]] auto approximate_svd_scratch(int rows, int rank, int batch_size) -> GesvdaScratch;
    auto approximate_svd(
        CuMatrixBatchedCF32Const input,
        void* scratch,
        const GesvdaScratch& layout,
        int* status,
        int batch_size
    ) -> void;
    [[nodiscard]] auto jacobi_svd_scratch(int rows, int rank, int batch_size) -> GesvdjScratch;
    auto jacobi_svd_batched(
        CuMatrixBatchedCF32 input, void* scratch, const GesvdjScratch& layout, int batch_size
    ) -> void;
    [[nodiscard]] auto svd_workspace() -> DeviceBuffer&;

  private:
    friend auto make_linalg(cudaStream_t stream) -> std::unique_ptr<Linalg>;

    Linalg(
        int device,
        cudaStream_t stream,
        CublasHandle&& blas,
        CusolverDnHandle&& solver,
        std::shared_ptr<DeviceArena> device_arena
    )
        : blas_(std::move(blas)), solver_(std::move(solver)), device_(device), stream_(stream),
          device_arena_(std::move(device_arena))
    {
    }

    Linalg(const Linalg&) = delete;
    Linalg(Linalg&&) = delete;
    auto operator=(const Linalg&) -> Linalg& = delete;
    auto operator=(Linalg&&) -> Linalg& = delete;
    auto destroy() -> void;

    std::optional<CublasHandle> blas_{};
    std::optional<CusolverDnHandle> solver_{};
    int device_{-1};
    cudaStream_t stream_{};
    std::shared_ptr<DeviceArena> device_arena_{};
    DeviceBuffer svd_workspace_{};
    gesvdjInfo_t gesvdj_parameters_{};
    std::map<std::tuple<int, int, int>, int> gesvda_workspace_counts_{};
    std::map<std::tuple<int, int, int>, int> gesvdj_workspace_counts_{};
};

static_assert(not std::is_default_constructible_v<Linalg>);
static_assert(not std::is_copy_constructible_v<Linalg>);
static_assert(not std::is_move_constructible_v<Linalg>);
}

#include "linalg/eo_ops.hpp"
#include "linalg/matmul.hpp"
#include "linalg/factory.cuh"
#include "linalg/lifecycle.cuh"
#include "linalg/state.cuh"

#endif
