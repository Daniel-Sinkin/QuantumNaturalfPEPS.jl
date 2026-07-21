#include "cuda_utils.cuh"
#include "linalg.cuh"
#include "sampler/kernels.cuh"

namespace qnpeps
{
auto batched_rangefinder(Linalg& la, const RangefinderArgs& args) -> void
{
    const auto input = args.input;
    const auto rows = input.rows();
    const auto cols = input.cols();
    const auto rank = args.rank;
    const auto dim_batch = args.dim_batch;
    const CuMatrixConstBatched omega_matrix{args.omega, 0, cols, rank};
    const CuMatrixBatched sketch_matrix{args.sketch, rows, rank};
    const CuMatrixBatched projection_matrix{args.projection, cols, rank};
    const CuMatrixBatched gram_matrix{args.gram, rank, rank};

    la.matmul_batched(input, omega_matrix, sketch_matrix, dim_batch);
    la.matmul_batched_left_adj(input, sketch_matrix, projection_matrix, dim_batch);
    la.matmul_batched(input, projection_matrix, sketch_matrix, dim_batch);

    for (auto pass = 0; pass < 2; ++pass)
    {
        la.matmul_batched_left_adj(sketch_matrix, sketch_matrix, gram_matrix, dim_batch);
        cu_chol_shift<<<grid_blocks_exact(dim_batch), k_threads_per_block, 0, la.stream()>>>(
            args.gram.p, rank, args.gram.stride, dim_batch
        );
        la.cholesky_lower_batched(rank, args.gram_ptrs, rank, args.info, dim_batch);
        if (args.fail_flag)
        {
            cu_any_chol_failed<<<
                grid_blocks_exact(dim_batch),
                k_threads_per_block,
                0,
                la.stream()>>>(args.info, dim_batch, args.fail_flag);
        }
        la.triangular_solve_batched(
            args.gram_ptrs,
            rank,
            args.sketch_ptrs,
            rows,
            rows,
            rank,
            dim_batch,
            {.op = BlasOp::conj_trans}
        );
    }

    const auto destination = args.q_out.data();
    const auto destination_stride = static_cast<usize>(args.q_out.stride());
    const auto destination_stride_bytes = destination_stride * sizeof(cuFloatComplex);
    const auto* source = args.sketch.p;
    const auto source_stride = static_cast<usize>(args.sketch.stride);
    const auto source_stride_bytes = source_stride * sizeof(cuFloatComplex);
    const auto matrix_elements = static_cast<usize>(rows) * static_cast<usize>(rank);
    const auto copy_width = matrix_elements * sizeof(cuFloatComplex);
    const auto copy_height = static_cast<usize>(dim_batch);
    CUDA_CHECK(cudaMemcpy2DAsync(
        destination,
        destination_stride_bytes,
        source,
        source_stride_bytes,
        copy_width,
        copy_height,
        cudaMemcpyDeviceToDevice,
        la.stream()
    ));
    la.matmul_batched_left_adj(sketch_matrix, input, args.r_out, dim_batch);
}
}
