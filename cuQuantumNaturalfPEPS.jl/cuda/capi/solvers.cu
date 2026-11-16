#include "capi/qnpeps.h"
#include "core/arena_cursor.cuh"
#include "core/cuda_utils.cuh"
#include "core/defer.cuh"
#include "core/session.cuh"
#include "gram/gram.cuh"
#include "linalg/linalg.cuh"
#include "linalg/rangefinder_rng.cuh"
#include "linalg/transfer.cuh"
#include "minsr/minsr.cuh"

#include <cstdint>
#include <span>
#include <vector>

using namespace qnpeps;

namespace
{
[[nodiscard]] auto rangefinder_scratch_bytes(int rows, int cols, int rank, int batch) -> int64_t
{
    const auto valid =
        rows >= 1 and cols >= 1 and rank >= 1 and batch >= 1 and rank <= rows and rank <= cols;
    require(
        valid,
        Status::bad_config,
        "rangefinder rejects a {}x{} input at rank {} over {} lanes",
        rows,
        cols,
        rank,
        batch
    );
    const auto rows_size = static_cast<usize>(rows);
    const auto cols_size = static_cast<usize>(cols);
    const auto rank_size = static_cast<usize>(rank);
    const auto batch_size = static_cast<usize>(batch);
    const auto complex_bytes = sizeof(cuFloatComplex);
    const auto bytes = device_align(complex_bytes * rows_size * rank_size * batch_size)
                       + device_align(complex_bytes * cols_size * rank_size * batch_size)
                       + device_align(complex_bytes * rank_size * rank_size * batch_size)
                       + device_align(complex_bytes * cols_size * rank_size)
                       + 2 * device_align(sizeof(cuFloatComplex*) * batch_size)
                       + device_align(sizeof(int) * batch_size) + device_align(sizeof(int))
                       + k_device_malloc_align;
    return static_cast<int64_t>(bytes);
}

struct RangefinderCall
{
    const void* input;
    int rows;
    int cols;
    int rank;
    int batch;
    int64_t input_stride;
    uint64_t seed;
    void* q_out;
    int64_t q_stride;
    void* r_out;
    int64_t r_stride;
    void* scratch;
    uint64_t scratch_bytes;
    void* stream;
};

auto batched_rangefinder_call(const RangefinderCall& call) -> void
{
    const auto has_buffers = call.input and call.q_out and call.r_out and call.scratch;
    require(has_buffers, Status::null_argument, "rangefinder misses a buffer");
    const auto rows = call.rows;
    const auto cols = call.cols;
    const auto rank = call.rank;
    const auto batch = call.batch;
    const auto valid_shape =
        rows >= 1 and cols >= 1 and rank >= 1 and batch >= 1 and rank <= rows and rank <= cols;
    require(
        valid_shape,
        Status::bad_config,
        "rangefinder rejects a {}x{} input at rank {} over {} lanes",
        rows,
        cols,
        rank,
        batch
    );

    const auto rows_i64 = static_cast<i64>(rows);
    const auto cols_i64 = static_cast<i64>(cols);
    const auto rank_i64 = static_cast<i64>(rank);
    const auto batch_i64 = static_cast<i64>(batch);
    const auto batch_size = static_cast<usize>(batch_i64);
    const auto omega_size = static_cast<usize>(cols_i64 * rank_i64);
    const auto invalid_stride = call.input_stride < rows_i64 * cols_i64
                                or call.q_stride < rows_i64 * rank_i64
                                or call.r_stride < rank_i64 * cols_i64;
    require(
        not invalid_stride,
        Status::bad_config,
        "rangefinder strides {}, {}, {} are below the matrix sizes",
        call.input_stride,
        call.q_stride,
        call.r_stride
    );

    auto cursor = ArenaCursor::carve(call.scratch, static_cast<usize>(call.scratch_bytes));
    auto* device_sketch =
        cursor.take<cuFloatComplex>(static_cast<usize>(rows_i64 * rank_i64 * batch_i64));
    auto* device_projection =
        cursor.take<cuFloatComplex>(static_cast<usize>(cols_i64 * rank_i64 * batch_i64));
    auto* device_gram =
        cursor.take<cuFloatComplex>(static_cast<usize>(rank_i64 * rank_i64 * batch_i64));
    auto* device_omega = cursor.take<cuFloatComplex>(omega_size);
    auto* device_gram_pointers = cursor.take<cuFloatComplex*>(batch_size);
    auto* device_sketch_pointers = cursor.take<cuFloatComplex*>(batch_size);
    auto* device_info = cursor.take<int>(batch_size);
    auto* device_fail_flag = cursor.take<int>(1);
    auto session = make_session(static_cast<cudaStream_t>(call.stream));
    auto& linalg = session->linalg();
    const auto cuda_stream = linalg.stream();

    auto rng = RangefinderRng::from_seed_and_width(call.seed, cols);
    std::vector<cuFloatComplex> host_omega{};
    host_omega.resize(omega_size);
    rng.fill_complex_normal(std::span{host_omega});

    std::vector<cuFloatComplex*> host_gram_pointers{};
    host_gram_pointers.resize(batch_size);
    std::vector<cuFloatComplex*> host_sketch_pointers{};
    host_sketch_pointers.resize(batch_size);
    for (auto lane = 0_i64; lane < batch_i64; ++lane)
    {
        const auto lane_index = static_cast<usize>(lane);
        host_gram_pointers[lane_index] = device_gram + lane * rank_i64 * rank_i64;
        host_sketch_pointers[lane_index] = device_sketch + lane * rows_i64 * rank_i64;
    }
    {
        auto uploaded = false;
        const Defer drain_after_failure{[&]
                                        {
                                            if (not uploaded)
                                                unchecked(cudaStreamSynchronize(cuda_stream));
                                        }};
        upload_async(linalg, device_omega, host_omega.data(), host_omega.size());
        upload_async(
            linalg, device_gram_pointers, host_gram_pointers.data(), host_gram_pointers.size()
        );
        upload_async(
            linalg, device_sketch_pointers, host_sketch_pointers.data(), host_sketch_pointers.size()
        );
        uploaded = true;
    }
    zero_async(linalg, device_fail_flag, 1);
    batched_rangefinder(
        linalg,
        {
            .input =
                CuMatrixBatchedCF32Const{
                    static_cast<const cuFloatComplex*>(call.input), call.input_stride, rows, cols
                },
            .rank = rank,
            .omega = device_omega,
            .q_out =
                CuMatrixBatchedCF32{
                    static_cast<cuFloatComplex*>(call.q_out), call.q_stride, rows, rank
                },
            .r_out =
                CuMatrixBatchedCF32{
                    static_cast<cuFloatComplex*>(call.r_out), call.r_stride, rank, cols
                },
            .dim_batch = batch,
            .sketch = CuSpanCF32{device_sketch, rows_i64 * rank_i64},
            .projection = CuSpanCF32{device_projection, cols_i64 * rank_i64},
            .gram = CuSpanCF32{device_gram, rank_i64 * rank_i64},
            .gram_ptrs = device_gram_pointers,
            .sketch_ptrs = device_sketch_pointers,
            .info = device_info,
            .fail_flag = device_fail_flag,
        }
    );
    check(cudaStreamSynchronize(linalg.stream()));
    int host_fail_flag{};
    download(&host_fail_flag, device_fail_flag, 1);
    require(
        host_fail_flag == 0,
        Status::cuda,
        "rangefinder Cholesky reported failure flag {} over {} lanes",
        host_fail_flag,
        batch
    );
}
}

extern "C" int64_t qnpeps_minsr_dense_count(const QnpepsMinsrDesc* descriptor)
{
    return guarded_or(
        int64_t{-1}, [&] { return qnpeps::minsr::descriptor_dense_count(descriptor); }
    );
}

extern "C" int64_t qnpeps_minsr_compact_count(const QnpepsMinsrDesc* descriptor)
{
    return guarded_or(
        int64_t{-1}, [&] { return qnpeps::minsr::descriptor_compact_count(descriptor); }
    );
}

extern "C" int64_t qnpeps_minsr_scratch_bytes(const QnpepsMinsrDesc* descriptor)
{
    return guarded_or(
        int64_t{-1}, [&] { return qnpeps::minsr::descriptor_scratch_bytes(descriptor); }
    );
}

extern "C" qnpeps_status qnpeps_minsr(
    const QnpepsMinsrDesc* descriptor, const QnpepsMinsrArgs* args
)
{
    return guarded([&] { qnpeps::minsr::execute(descriptor, args); });
}

extern "C" qnpeps_status qnpeps_minsr_ctx_create(
    const QnpepsMinsrDesc* descriptor, void* stream, qnpeps_minsr_ctx** out
)
{
    return guarded([&] { qnpeps::minsr::ctx_create(descriptor, stream, out); });
}

extern "C" qnpeps_status qnpeps_minsr_ctx_run(qnpeps_minsr_ctx* ctx, const QnpepsMinsrArgs* args)
{
    return guarded([&] { qnpeps::minsr::ctx_run(ctx, args); });
}

extern "C" void qnpeps_minsr_ctx_destroy(qnpeps_minsr_ctx* ctx)
{
    static_cast<void>(guarded_keeping_message([&] { qnpeps::minsr::ctx_destroy(ctx); }));
}

extern "C" qnpeps_status qnpeps_gram_ctx_create(
    const QnpepsGramDesc* descriptor, void* stream, qnpeps_gram_ctx** out
)
{
    return guarded([&] { qnpeps::gram::ctx_create(descriptor, stream, out); });
}

extern "C" qnpeps_status qnpeps_gram_ctx_run(qnpeps_gram_ctx* ctx, const QnpepsGramArgs* args)
{
    return guarded([&] { qnpeps::gram::ctx_run(ctx, args); });
}

extern "C" qnpeps_status qnpeps_gram_ctx_footprint(
    const qnpeps_gram_ctx* ctx, QnpepsGramFootprint* out
)
{
    return guarded([&] { qnpeps::gram::ctx_footprint(ctx, out); });
}

extern "C" void qnpeps_gram_ctx_destroy(qnpeps_gram_ctx* ctx)
{
    static_cast<void>(guarded_keeping_message([&] { qnpeps::gram::ctx_destroy(ctx); }));
}

extern "C" int64_t qnpeps_batched_rangefinder_scratch_bytes(int rows, int cols, int rank, int batch)
{
    return guarded_or(
        int64_t{-1}, [&] { return rangefinder_scratch_bytes(rows, cols, rank, batch); }
    );
}

extern "C" qnpeps_status qnpeps_batched_rangefinder(
    const void* input,
    int rows,
    int cols,
    int rank,
    int batch,
    int64_t input_stride,
    uint64_t seed,
    void* q_out,
    int64_t q_stride,
    void* r_out,
    int64_t r_stride,
    void* scratch,
    uint64_t scratch_bytes,
    void* stream
)
{
    return guarded(
        [&]
        {
            batched_rangefinder_call({
                input,
                rows,
                cols,
                rank,
                batch,
                input_stride,
                seed,
                q_out,
                q_stride,
                r_out,
                r_stride,
                scratch,
                scratch_bytes,
                stream,
            });
        }
    );
}
