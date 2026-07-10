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
    const auto k = args.k;
    const auto dim_batch = args.dim_batch;
    const CuMatrixConstBatched mat_omega{args.omega, 0, cols, k};
    const CuMatrixBatched sketch_mat{args.sketch, rows, k};
    const CuMatrixBatched proj_mat{args.proj, cols, k};
    const CuMatrixBatched gram_mat{args.gram, k, k};

    la.matmul_batched(input, mat_omega, sketch_mat, dim_batch);
    la.matmul_batched_adj_none(input, sketch_mat, proj_mat, dim_batch);
    la.matmul_batched(input, proj_mat, sketch_mat, dim_batch);

    for (auto pass = 0; pass < 2; ++pass)
    {
        la.matmul_batched_adj_none(sketch_mat, sketch_mat, gram_mat, dim_batch);
        cu_chol_shift<<<grid_blocks_exact(dim_batch), k_threads_per_block, 0, la.stream()>>>(
            args.gram.p, k, args.gram.stride, dim_batch
        );
        la.cholesky_lower_batched(k, args.gram_ptrs, k, args.info, dim_batch);
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
            k,
            args.sketch_ptrs,
            rows,
            rows,
            k,
            dim_batch,
            {.op = BlasOp::conj_trans}
        );
    }

    const auto dst = args.q_out.data();
    const auto dst_stride_bytes = static_cast<usize>(args.q_out.stride()) * sizeof(cf);
    const auto* src = args.sketch.p;
    const auto src_stride_bytes = static_cast<usize>(args.sketch.stride) * sizeof(cf);
    const auto copy_width = static_cast<usize>(rows) * static_cast<usize>(k) * sizeof(cf);
    const auto copy_height = static_cast<usize>(dim_batch);
    CUDA_CHECK(cudaMemcpy2DAsync(
        dst,
        dst_stride_bytes,
        src,
        src_stride_bytes,
        copy_width,
        copy_height,
        cudaMemcpyDeviceToDevice,
        la.stream()
    ));
    la.matmul_batched_adj_none(sketch_mat, input, args.r_out, dim_batch);
}
}
