#include "core/arena_cursor.cuh"
#include "core/cuda_utils.cuh"
#include "core/defer.cuh"
#include "core/error.cuh"
#include "core/session.cuh"
#include "gram/gram.cuh"
#include "gram/gram_cublas.cuh"
#include "gram/gram_slab.cuh"
#include "linalg/linalg.cuh"
#include "minsr/ok_layout.cuh"

#include <algorithm>
#include <array>
#include <climits>
#include <limits>
#include <mutex>
#include <new>
#include <vector>

namespace
{
using qnpeps::cublas_status;
using qnpeps::cuda_status;
using qnpeps::Dims;
using qnpeps::i64;
using qnpeps::u64;
using qnpeps::usize;

constexpr i64 k_row_chunk_bytes{static_cast<i64>(1) << 30};

[[nodiscard]] auto checked_bytes(u64 elements, u64 element_size, u64& bytes) -> bool
{
    if (elements != 0 and element_size > std::numeric_limits<u64>::max() / elements) return false;
    bytes = elements * element_size;
    return bytes <= std::numeric_limits<usize>::max();
}

[[nodiscard]] auto save_blas_state(qnpeps::Linalg& linalg, qnpeps::BlasState& state)
    -> qnpeps_status
{
    return qnpeps::cublas_status(linalg.blas_state(state));
}

auto restore_blas_state(qnpeps::Linalg& linalg, const qnpeps::BlasState& state) -> void
{
    CUBLAS_CHECK(linalg.set_blas_state(state));
}

auto julia_linalg(void* handle, cudaStream_t stream) -> qnpeps::Linalg*
{
    if (not handle) return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG), nullptr;
    auto& linalg = static_cast<qnpeps::Session*>(handle)->linalg();
    if (stream and stream != linalg.stream())
        return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG), nullptr;
    return &linalg;
}

auto establish_julia_blas_state(qnpeps::Linalg& linalg) -> qnpeps_status
{
    return qnpeps::cublas_status(linalg.set_gram_state());
}

extern "C" __attribute__((visibility("default"))) qnpeps_status
qnpeps_julia_blas_create(void* stream, void** out)
{
    qnpeps::reset_err();
    if (not out) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    *out = nullptr;
    auto session = qnpeps::make_session(static_cast<cudaStream_t>(stream));
    if (not session) return qnpeps::err_state();
    const auto status = session->linalg().set_gram_state();
    if (status != CUBLAS_STATUS_SUCCESS) return qnpeps::cublas_status(status);
    *out = session.release();
    return QNPEPS_OK;
}

extern "C" __attribute__((visibility("default"))) qnpeps_status
qnpeps_julia_blas_destroy(void* handle)
{
    qnpeps::reset_err();
    if (not handle) return QNPEPS_OK;
    delete static_cast<qnpeps::Session*>(handle);
    return QNPEPS_OK;
}

extern "C" __attribute__((visibility("default"))) qnpeps_status qnpeps_julia_gram_expand(
    void* dense,
    const void* rows,
    const void* samples,
    int64_t row_count,
    int64_t compact,
    int64_t sample_base,
    int sites,
    int first_site,
    int compact_begin,
    int compact_count,
    int dense_width,
    const int* slot,
    const int* offsets,
    const int* slices,
    void* stream
)
{
    qnpeps::reset_err();
    return qnpeps::cuda_status(
        qnpeps::gram_cublas::expand(
            static_cast<cuFloatComplex*>(dense),
            static_cast<const cuFloatComplex*>(rows),
            static_cast<const std::uint8_t*>(samples),
            row_count,
            compact,
            sample_base,
            sites,
            first_site,
            compact_begin,
            compact_count,
            dense_width,
            slot,
            offsets,
            slices,
            static_cast<cudaStream_t>(stream)
        )
    );
}

extern "C" __attribute__((visibility("default"))) qnpeps_status qnpeps_julia_gram_herk(
    void* handle,
    int rows,
    int dense_width,
    const void* dense,
    void* output,
    int output_ld,
    void* stream
)
{
    qnpeps::reset_err();
    auto* linalg = julia_linalg(handle, static_cast<cudaStream_t>(stream));
    if (not linalg) return qnpeps::err_state();
    const auto state_status = establish_julia_blas_state(*linalg);
    if (state_status != QNPEPS_OK) return state_status;
    return qnpeps::cublas_status(
        qnpeps::gram_cublas::accumulate_diagonal_block(
            *linalg,
            rows,
            dense_width,
            static_cast<const cuFloatComplex*>(dense),
            static_cast<cuFloatComplex*>(output),
            output_ld
        )
    );
}

extern "C" __attribute__((visibility("default"))) qnpeps_status qnpeps_julia_gram_gemm(
    void* handle,
    int rows_a,
    int rows_b,
    int dense_width,
    const void* dense_a,
    const void* dense_b,
    void* output,
    int output_ld,
    void* stream
)
{
    qnpeps::reset_err();
    auto* linalg = julia_linalg(handle, static_cast<cudaStream_t>(stream));
    if (not linalg) return qnpeps::err_state();
    const auto state_status = establish_julia_blas_state(*linalg);
    if (state_status != QNPEPS_OK) return state_status;
    return qnpeps::cublas_status(
        qnpeps::gram_cublas::accumulate_offdiagonal_block(
            *linalg,
            rows_a,
            rows_b,
            dense_width,
            static_cast<const cuFloatComplex*>(dense_a),
            static_cast<const cuFloatComplex*>(dense_b),
            static_cast<cuFloatComplex*>(output),
            output_ld
        )
    );
}

extern "C" __attribute__((visibility("default"))) qnpeps_status qnpeps_julia_gram_finalize(
    void* output, int output_ld, int row_count, int local_base, int samples, void* stream
)
{
    qnpeps::reset_err();
    return qnpeps::cuda_status(
        qnpeps::gram_cublas::finalize(
            static_cast<cuFloatComplex*>(output),
            output_ld,
            row_count,
            local_base,
            samples,
            static_cast<cudaStream_t>(stream)
        )
    );
}

[[nodiscard]] auto check_descriptor(const QnpepsGramDesc* descriptor) -> qnpeps_status
{
    if (not descriptor) return QNPEPS_ERR_NULL_ARG;
    if (descriptor->struct_size != sizeof(QnpepsGramDesc)) return QNPEPS_ERR_BAD_VERSION;
    if (descriptor->reserved != 0) return QNPEPS_ERR_BAD_CONFIG;
    const auto invalid_dimensions = descriptor->lx < 2 or descriptor->ly < 2
                                    or descriptor->dim_phys != 2 or descriptor->dim_bond < 1;
    if (invalid_dimensions) return QNPEPS_ERR_BAD_CONFIG;
    if (descriptor->consumer != QNPEPS_GRAM_CONSUMER_SLAB) return QNPEPS_ERR_BAD_CONFIG;
    if (descriptor->n_samples < 2 or descriptor->n_samples > INT_MAX) return QNPEPS_ERR_BAD_CONFIG;
    return QNPEPS_OK;
}

struct VirtualShard
{
    i64 base{};
    i64 count{};
};

[[nodiscard]] auto virtual_shards(i64 samples)
    -> std::array<VirtualShard, qnpeps::gram_slab::k_virtual_shards>
{
    std::array<VirtualShard, qnpeps::gram_slab::k_virtual_shards> shards{};
    const i64 quotient{samples / qnpeps::gram_slab::k_virtual_shards};
    const i64 remainder{samples % qnpeps::gram_slab::k_virtual_shards};
    i64 base{};
    for (int index{}; index < qnpeps::gram_slab::k_virtual_shards; ++index)
    {
        shards[static_cast<usize>(index)].base = base;
        shards[static_cast<usize>(index)].count = quotient + (index < remainder ? 1 : 0);
        base += shards[static_cast<usize>(index)].count;
    }
    return shards;
}
}

struct qnpeps_gram_ctx
{
    [[nodiscard]] auto linalg() noexcept -> qnpeps::Linalg& { return *execution; }
    [[nodiscard]] auto stream() const noexcept -> cudaStream_t { return execution->stream(); }

    QnpepsGramDesc desc{};
    i64 sites{};
    i64 compact{};
    int max_slice{};
    int device{};
    std::unique_ptr<qnpeps::Session> session{};
    qnpeps::Linalg* execution{};
    std::vector<int> offsets{};
    std::vector<int> slices{};
    std::vector<int> slot{};
    int* offsets_device{};
    int* slices_device{};
    int* slot_device{};
    cuFloatComplex* dense_a{};
    cuFloatComplex* dense_b{};
    i64 slab_width{};
    i64 virtual_rows{};
    i64 chunk_rows{};
    QnpepsGramFootprint footprint{};
    std::mutex mutex{};
};

namespace
{
[[nodiscard]] auto fill_geometry(qnpeps_gram_ctx& context) -> qnpeps_status
{
    const Dims dims{context.desc.lx, context.desc.ly, context.desc.dim_phys, context.desc.dim_bond};
    const auto sites = static_cast<usize>(context.sites);
    const auto compact = static_cast<usize>(context.compact);
    context.offsets.reserve(sites);
    context.slices.reserve(sites);
    context.slot.reserve(compact);
    i64 offset{};
    int site{};
    for (int row{}; row < context.desc.lx; ++row)
    {
        for (int column{}; column < context.desc.ly; ++column)
        {
            const i64 slice{qnpeps::minsr::site_slice(dims, row, column)};
            if (slice < 1 or slice > INT_MAX or offset > INT_MAX) return QNPEPS_ERR_BAD_CONFIG;
            context.offsets.push_back(static_cast<int>(offset));
            context.slices.push_back(static_cast<int>(slice));
            context.max_slice = std::max(context.max_slice, static_cast<int>(slice));
            for (i64 local{}; local < slice; ++local)
                context.slot.push_back(site);
            offset += slice;
            ++site;
        }
    }
    return offset == context.compact and site == context.sites ? QNPEPS_OK : QNPEPS_ERR_INTERNAL;
}

auto carve_context(qnpeps::ArenaCursor& cursor, qnpeps_gram_ctx& context) -> void
{
    const auto sites = static_cast<usize>(context.sites);
    const auto compact = static_cast<usize>(context.compact);
    const auto dense_a =
        static_cast<usize>(context.virtual_rows) * static_cast<usize>(context.slab_width);
    const auto dense_b =
        static_cast<usize>(context.chunk_rows) * static_cast<usize>(context.slab_width);
    context.offsets_device = cursor.take<int>(sites);
    context.slices_device = cursor.take<int>(sites);
    context.slot_device = cursor.take<int>(compact);
    context.dense_a = cursor.take<cuFloatComplex>(dense_a);
    context.dense_b = cursor.take<cuFloatComplex>(dense_b);
}

[[nodiscard]] auto upload_geometry(qnpeps_gram_ctx& context) -> qnpeps_status
{
    const auto sites = static_cast<usize>(context.sites);
    const auto compact = static_cast<usize>(context.compact);
    qnpeps_status status{cuda_status(cudaMemcpyAsync(
        context.offsets_device,
        context.offsets.data(),
        sizeof(int) * sites,
        cudaMemcpyHostToDevice,
        context.stream()
    ))};
    if (status == QNPEPS_OK)
    {
        status = cuda_status(cudaMemcpyAsync(
            context.slices_device,
            context.slices.data(),
            sizeof(int) * sites,
            cudaMemcpyHostToDevice,
            context.stream()
        ));
    }
    if (status == QNPEPS_OK)
    {
        status = cuda_status(cudaMemcpyAsync(
            context.slot_device,
            context.slot.data(),
            sizeof(int) * compact,
            cudaMemcpyHostToDevice,
            context.stream()
        ));
    }
    if (status == QNPEPS_OK) status = cuda_status(cudaStreamSynchronize(context.stream()));
    return status;
}

[[nodiscard]] auto build_footprint(qnpeps_gram_ctx& context, u64 arena_bytes) -> qnpeps_status
{
    QnpepsGramFootprint footprint{};
    footprint.struct_size = sizeof(footprint);
    u64 sites_bytes{};
    u64 slot_bytes{};
    u64 dense_a_bytes{};
    u64 dense_b_bytes{};
    u64 samples_bytes{};
    u64 rows_bytes{};
    u64 gram_bytes{};
    auto valid_bytes = checked_bytes(static_cast<u64>(context.sites), 2 * sizeof(int), sites_bytes);
    if (valid_bytes)
        valid_bytes = checked_bytes(static_cast<u64>(context.compact), sizeof(int), slot_bytes);
    if (valid_bytes)
        valid_bytes = checked_bytes(
            static_cast<u64>(context.virtual_rows) * static_cast<u64>(context.slab_width),
            sizeof(cuFloatComplex),
            dense_a_bytes
        );
    if (valid_bytes)
        valid_bytes = checked_bytes(
            static_cast<u64>(context.chunk_rows) * static_cast<u64>(context.slab_width),
            sizeof(cuFloatComplex),
            dense_b_bytes
        );
    if (valid_bytes)
        valid_bytes = checked_bytes(
            static_cast<u64>(context.desc.n_samples) * static_cast<u64>(context.sites),
            sizeof(std::uint8_t),
            samples_bytes
        );
    if (valid_bytes)
        valid_bytes = checked_bytes(
            static_cast<u64>(context.desc.n_samples) * static_cast<u64>(context.compact),
            sizeof(cuFloatComplex),
            rows_bytes
        );
    if (valid_bytes)
        valid_bytes = checked_bytes(
            static_cast<u64>(context.desc.n_samples) * static_cast<u64>(context.desc.n_samples),
            sizeof(cuFloatComplex),
            gram_bytes
        );
    if (not valid_bytes) return QNPEPS_ERR_BAD_CONFIG;
    footprint.geometry_device_bytes = sites_bytes + slot_bytes;
    footprint.dense_a_device_bytes = dense_a_bytes;
    footprint.dense_b_device_bytes = dense_b_bytes;
    footprint.context_device_bytes = arena_bytes;
    footprint.caller_samples_bytes = samples_bytes;
    footprint.caller_rows_bytes = rows_bytes;
    footprint.caller_gram_bytes = gram_bytes;
    context.footprint = footprint;
    return QNPEPS_OK;
}

[[nodiscard]] auto check_args(const qnpeps_gram_ctx& context, const QnpepsGramArgs* args)
    -> qnpeps_status
{
    if (not args) return QNPEPS_ERR_NULL_ARG;
    if (args->struct_size != sizeof(QnpepsGramArgs)) return QNPEPS_ERR_BAD_VERSION;
    if (args->reserved != 0) return QNPEPS_ERR_BAD_CONFIG;
    if (not args->samples or not args->o_rows or not args->gram_out) return QNPEPS_ERR_NULL_ARG;
    if (args->stream and static_cast<cudaStream_t>(args->stream) != context.stream())
        return QNPEPS_ERR_BAD_CONFIG;
    const auto insufficient_bytes = args->samples_bytes < context.footprint.caller_samples_bytes
                                    or args->o_rows_bytes < context.footprint.caller_rows_bytes
                                    or args->gram_out_bytes < context.footprint.caller_gram_bytes;
    if (insufficient_bytes) return QNPEPS_ERR_BAD_CONFIG;
    return QNPEPS_OK;
}

[[nodiscard]] auto validate_schedule(qnpeps_gram_ctx& context) -> qnpeps_status
{
    const qnpeps::gram_slab::Geometry geometry{
        context.sites,
        context.compact,
        context.offsets.data(),
        context.slices.data(),
        context.slot_device,
        context.offsets_device,
        context.slices_device
    };
    const int slabs{qnpeps::gram_slab::slab_count(geometry, context.slab_width)};
    int virtual_shard_count{0};
    int block_calls{0};
    const auto shards{virtual_shards(context.desc.n_samples)};
    for (const VirtualShard& destination : shards)
    {
        if (destination.count < 1) continue;
        ++virtual_shard_count;
        for (const VirtualShard& source : shards)
        {
            if (source.count < 1) continue;
            if (source.base == destination.base)
            {
                ++block_calls;
                continue;
            }
            block_calls +=
                static_cast<int>((source.count + context.chunk_rows - 1) / context.chunk_rows);
        }
    }
    if (slabs < 1 or virtual_shard_count < 1 or block_calls < 1) return QNPEPS_ERR_INTERNAL;
    return QNPEPS_OK;
}
}

namespace qnpeps::gram
{
auto ctx_create_impl(
    const QnpepsGramDesc* descriptor,
    std::unique_ptr<Session> session,
    Linalg& linalg,
    qnpeps_gram_ctx** out
) -> qnpeps_status
{
    auto* context = new (std::nothrow) qnpeps_gram_ctx;
    if (not context) return set_err(QNPEPS_ERR_OOM);
    context->desc = *descriptor;
    context->sites = static_cast<i64>(descriptor->lx) * descriptor->ly;
    context->compact = minsr::compact_count(
        Dims{descriptor->lx, descriptor->ly, descriptor->dim_phys, descriptor->dim_bond}
    );
    context->device = linalg.device();
    context->execution = &linalg;
    context->session = std::move(session);
    auto status = fill_geometry(*context);
    context->virtual_rows =
        (descriptor->n_samples + gram_slab::k_virtual_shards - 1) / gram_slab::k_virtual_shards;
    const i64 row_bytes{context->compact * static_cast<i64>(sizeof(cuFloatComplex))};
    const i64 visit_rows{std::min<i64>(
        descriptor->n_samples, std::max<i64>(1, k_row_chunk_bytes / std::max<i64>(row_bytes, 1))
    )};
    context->chunk_rows = std::min(context->virtual_rows, visit_rows);
    context->slab_width =
        gram_slab::default_width(context->compact, context->max_slice, descriptor->n_samples);

    usize context_arena_bytes{};
    if (status == QNPEPS_OK)
    {
        auto& context_arena = linalg.persistent_arena();
        const auto context_arena_start = context_arena.total();
        carve_context(context_arena, *context);
        if (err_state() != QNPEPS_OK) status = err_state();
        context_arena_bytes = context_arena.total() - context_arena_start;
    }
    if (status == QNPEPS_OK) status = upload_geometry(*context);
    if (status == QNPEPS_OK)
    {
        status = establish_julia_blas_state(context->linalg());
    }
    if (status == QNPEPS_OK) status = build_footprint(*context, context_arena_bytes);
    if (status == QNPEPS_OK) status = validate_schedule(*context);
    if (status != QNPEPS_OK)
    {
        ctx_destroy(context);
        return set_err(status);
    }
    *out = context;
    return QNPEPS_OK;
}

auto ctx_create(const QnpepsGramDesc* descriptor, void* stream, qnpeps_gram_ctx** out)
    -> qnpeps_status
{
    if (not out) return set_err(QNPEPS_ERR_NULL_ARG);
    *out = nullptr;
    const auto descriptor_status = check_descriptor(descriptor);
    if (descriptor_status != QNPEPS_OK) return set_err(descriptor_status);

    auto session = make_session(static_cast<cudaStream_t>(stream));
    if (not session) return err_state();
    auto& linalg = session->linalg();
    return ctx_create_impl(descriptor, std::move(session), linalg, out);
}

auto ctx_create(const QnpepsGramDesc* descriptor, Linalg& linalg, qnpeps_gram_ctx** out)
    -> qnpeps_status
{
    if (not out) return set_err(QNPEPS_ERR_NULL_ARG);
    *out = nullptr;
    const auto descriptor_status = check_descriptor(descriptor);
    if (descriptor_status != QNPEPS_OK) return set_err(descriptor_status);
    return ctx_create_impl(descriptor, nullptr, linalg, out);
}

auto ctx_run(qnpeps_gram_ctx* context, const QnpepsGramArgs* args) -> qnpeps_status
{
    if (not context) return set_err(QNPEPS_ERR_NULL_ARG);
    const std::lock_guard<std::mutex> lock{context->mutex};
    const auto args_status = check_args(*context, args);
    if (args_status != QNPEPS_OK) return set_err(args_status);

    int caller{};
    auto status = cuda_status(cudaGetDevice(&caller));
    if (status != QNPEPS_OK) return status;
    if (caller != context->device) return set_err(QNPEPS_ERR_BAD_CONFIG);

    qnpeps::BlasState blas_state{};
    status = save_blas_state(context->linalg(), blas_state);
    if (status != QNPEPS_OK) return status;
    DEFER([&] { restore_blas_state(context->linalg(), blas_state); });
    status = establish_julia_blas_state(context->linalg());
    if (status != QNPEPS_OK) return status;

    const auto samples = static_cast<i64>(context->desc.n_samples);
    auto* gram = static_cast<cuFloatComplex*>(args->gram_out);
    const auto* rows = static_cast<const cuFloatComplex*>(args->o_rows);
    const auto* device_samples = static_cast<const std::uint8_t*>(args->samples);
    status = cuda_status(cudaMemsetAsync(
        gram, 0, sizeof(cuFloatComplex) * static_cast<usize>(samples * samples), context->stream()
    ));

    const gram_slab::Geometry geometry{
        context->sites,
        context->compact,
        context->offsets.data(),
        context->slices.data(),
        context->slot_device,
        context->offsets_device,
        context->slices_device
    };
    const gram_slab::Workspace workspace{
        context->dense_a, context->dense_b, context->slab_width, &context->linalg()
    };
    const auto shards{virtual_shards(samples)};

    for (const VirtualShard& destination : shards)
    {
        if (destination.count < 1 or status != QNPEPS_OK) continue;
        auto rows_a{rows + destination.base * context->compact};
        auto blockrow{gram + destination.base * samples};
        for (const VirtualShard& source : shards)
        {
            if (source.count < 1 or status != QNPEPS_OK) continue;
            if (source.base == destination.base)
            {
                status = gram_slab::accumulate_block(
                    workspace,
                    geometry,
                    device_samples,
                    rows_a,
                    destination.base,
                    destination.count,
                    rows_a,
                    destination.base,
                    destination.count,
                    blockrow + destination.base,
                    static_cast<int>(samples),
                    true
                );
                if (status == QNPEPS_OK)
                    status = cuda_status(cudaStreamSynchronize(context->stream()));
                continue;
            }
            for (i64 local{}; local < source.count; local += context->chunk_rows)
            {
                if (status != QNPEPS_OK) break;
                const i64 count{std::min(context->chunk_rows, source.count - local)};
                auto rows_b{rows + (source.base + local) * context->compact};
                status = gram_slab::accumulate_block(
                    workspace,
                    geometry,
                    device_samples,
                    rows_a,
                    destination.base,
                    destination.count,
                    rows_b,
                    source.base + local,
                    count,
                    blockrow + source.base + local,
                    static_cast<int>(samples),
                    false
                );
                if (status == QNPEPS_OK)
                    status = cuda_status(cudaStreamSynchronize(context->stream()));
            }
        }
        if (status == QNPEPS_OK)
        {
            status = gram_slab::finalize_block(
                blockrow,
                static_cast<int>(samples),
                static_cast<int>(destination.count),
                static_cast<int>(destination.base),
                static_cast<int>(samples),
                context->stream()
            );
        }
        if (status == QNPEPS_OK) status = cuda_status(cudaStreamSynchronize(context->stream()));
    }
    if (status != QNPEPS_OK) return set_err(status);
    return QNPEPS_OK;
}

auto ctx_footprint(const qnpeps_gram_ctx* context, QnpepsGramFootprint* out) -> qnpeps_status
{
    if (not context or not out) return set_err(QNPEPS_ERR_NULL_ARG);
    if (out->struct_size != sizeof(QnpepsGramFootprint)) return set_err(QNPEPS_ERR_BAD_VERSION);
    *out = context->footprint;
    return QNPEPS_OK;
}

auto ctx_destroy(qnpeps_gram_ctx* context) -> void
{
    if (not context) return;
    int caller{};
    const bool have_caller{cudaGetDevice(&caller) == cudaSuccess};
    const bool switched{
        have_caller and caller != context->device and cudaSetDevice(context->device) == cudaSuccess
    };
    CUDA_NOCHECK(cudaStreamSynchronize(context->stream()));
    if (switched) CUDA_NOCHECK(cudaSetDevice(caller));
    delete context;
}
}
