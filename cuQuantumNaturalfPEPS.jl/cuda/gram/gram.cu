#include "capi/internal/format.cuh"
#include "core/arena_cursor.cuh"
#include "core/cuda_utils.cuh"
#include "core/defer.cuh"
#include "core/error.hpp"
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
#include <vector>

namespace
{
using qnpeps::check;
using qnpeps::Dims;
using qnpeps::i64;
using qnpeps::require;
using qnpeps::require_struct_size;
using qnpeps::Status;
using qnpeps::u64;
using qnpeps::usize;

constexpr i64 k_row_chunk_bytes{static_cast<i64>(1) << 30};

[[nodiscard]] auto checked_bytes(u64 elements, u64 element_size, u64& bytes) -> bool
{
    if (elements != 0 and element_size > std::numeric_limits<u64>::max() / elements) return false;
    bytes = elements * element_size;
    return bytes <= std::numeric_limits<usize>::max();
}

auto validate_descriptor(const QnpepsGramDesc* descriptor) -> void
{
    require(descriptor != nullptr, Status::null_argument, "Gram descriptor is null");
    require_struct_size(*descriptor, "Gram descriptor");
    require(
        descriptor->reserved == 0,
        Status::bad_config,
        "Gram descriptor reserved field is {}",
        descriptor->reserved
    );
    const auto invalid_dimensions = descriptor->lx < 2 or descriptor->ly < 2
                                    or descriptor->dim_phys != 2 or descriptor->dim_bond < 1;
    require(not invalid_dimensions, Status::bad_config, "Gram descriptor rejects {}", *descriptor);
    require(
        descriptor->consumer == QNPEPS_GRAM_CONSUMER_SLAB,
        Status::bad_config,
        "Gram descriptor is not for the slab consumer, {}",
        *descriptor
    );
    require(
        descriptor->n_samples >= 2 and descriptor->n_samples <= INT_MAX,
        Status::bad_config,
        "Gram sample count outside 2 to {}, {}",
        INT_MAX,
        *descriptor
    );
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
auto fill_geometry(qnpeps_gram_ctx& context) -> void
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
            const auto fits = slice >= 1 and slice <= INT_MAX and offset <= INT_MAX;
            require(
                fits,
                Status::bad_config,
                "Gram site ({}, {}) slice {} at offset {} exceeds the int range",
                row,
                column,
                slice,
                offset
            );
            context.offsets.push_back(static_cast<int>(offset));
            context.slices.push_back(static_cast<int>(slice));
            context.max_slice = std::max(context.max_slice, static_cast<int>(slice));
            for (i64 local{}; local < slice; ++local)
                context.slot.push_back(site);
            offset += slice;
            ++site;
        }
    }
    require(
        offset == context.compact and site == context.sites,
        "Gram geometry covers {} slots over {} sites, expected {} over {}",
        offset,
        site,
        context.compact,
        context.sites
    );
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

auto upload_geometry(qnpeps_gram_ctx& context) -> void
{
    const auto sites = static_cast<usize>(context.sites);
    const auto compact = static_cast<usize>(context.compact);
    check(cudaMemcpyAsync(
        context.offsets_device,
        context.offsets.data(),
        sizeof(int) * sites,
        cudaMemcpyHostToDevice,
        context.stream()
    ));
    check(cudaMemcpyAsync(
        context.slices_device,
        context.slices.data(),
        sizeof(int) * sites,
        cudaMemcpyHostToDevice,
        context.stream()
    ));
    check(cudaMemcpyAsync(
        context.slot_device,
        context.slot.data(),
        sizeof(int) * compact,
        cudaMemcpyHostToDevice,
        context.stream()
    ));
    check(cudaStreamSynchronize(context.stream()));
}

auto build_footprint(qnpeps_gram_ctx& context, u64 arena_bytes) -> void
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
    require(
        valid_bytes,
        Status::bad_config,
        "Gram footprint for {} samples overflows the byte range",
        context.desc.n_samples
    );
    footprint.geometry_device_bytes = sites_bytes + slot_bytes;
    footprint.dense_a_device_bytes = dense_a_bytes;
    footprint.dense_b_device_bytes = dense_b_bytes;
    footprint.context_device_bytes = arena_bytes;
    footprint.caller_samples_bytes = samples_bytes;
    footprint.caller_rows_bytes = rows_bytes;
    footprint.caller_gram_bytes = gram_bytes;
    context.footprint = footprint;
}

auto validate_args(const qnpeps_gram_ctx& context, const QnpepsGramArgs* args) -> void
{
    require(args != nullptr, Status::null_argument, "Gram arguments are null");
    require_struct_size(*args, "Gram arguments");
    require(
        args->reserved == 0,
        Status::bad_config,
        "Gram arguments reserved field is {}",
        args->reserved
    );
    const auto has_buffers = args->samples and args->o_rows and args->gram_out;
    require(has_buffers, Status::null_argument, "Gram arguments miss a buffer");
    const auto foreign_stream =
        args->stream and static_cast<cudaStream_t>(args->stream) != context.stream();
    require(
        not foreign_stream, Status::bad_config, "Gram run stream differs from the context stream"
    );
    const auto insufficient_bytes = args->samples_bytes < context.footprint.caller_samples_bytes
                                    or args->o_rows_bytes < context.footprint.caller_rows_bytes
                                    or args->gram_out_bytes < context.footprint.caller_gram_bytes;
    require(
        not insufficient_bytes,
        Status::bad_config,
        "Gram buffers hold {}, {}, {} bytes, need {}, {}, {}",
        args->samples_bytes,
        args->o_rows_bytes,
        args->gram_out_bytes,
        context.footprint.caller_samples_bytes,
        context.footprint.caller_rows_bytes,
        context.footprint.caller_gram_bytes
    );
}

auto validate_schedule(qnpeps_gram_ctx& context) -> void
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
    require(
        slabs >= 1 and virtual_shard_count >= 1 and block_calls >= 1,
        "Gram schedule has {} slabs, {} shards, {} block calls",
        slabs,
        virtual_shard_count,
        block_calls
    );
}
}

namespace qnpeps::gram
{
auto ctx_create_impl(
    const QnpepsGramDesc* descriptor,
    std::unique_ptr<Session> session,
    Linalg& linalg,
    qnpeps_gram_ctx** out
) -> void
{
    auto* context = new qnpeps_gram_ctx;
    context->device = linalg.device();
    context->execution = &linalg;
    context->session = std::move(session);
    auto created = false;
    const Defer destroy_after_failure{[&]
                                      {
                                          if (not created) ctx_destroy(context);
                                      }};
    context->desc = *descriptor;
    context->sites = static_cast<i64>(descriptor->lx) * descriptor->ly;
    context->compact = minsr::compact_count(
        Dims{descriptor->lx, descriptor->ly, descriptor->dim_phys, descriptor->dim_bond}
    );
    fill_geometry(*context);
    context->virtual_rows =
        (descriptor->n_samples + gram_slab::k_virtual_shards - 1) / gram_slab::k_virtual_shards;
    const i64 row_bytes{context->compact * static_cast<i64>(sizeof(cuFloatComplex))};
    const i64 visit_rows{std::min<i64>(
        descriptor->n_samples, std::max<i64>(1, k_row_chunk_bytes / std::max<i64>(row_bytes, 1))
    )};
    context->chunk_rows = std::min(context->virtual_rows, visit_rows);
    context->slab_width =
        gram_slab::default_width(context->compact, context->max_slice, descriptor->n_samples);

    auto& context_arena = linalg.persistent_arena();
    const auto context_arena_start = context_arena.total();
    carve_context(context_arena, *context);
    const auto context_arena_bytes = context_arena.total() - context_arena_start;
    upload_geometry(*context);
    context->linalg().set_gram_state();
    build_footprint(*context, context_arena_bytes);
    validate_schedule(*context);
    *out = context;
    created = true;
}

auto ctx_create(const QnpepsGramDesc* descriptor, void* stream, qnpeps_gram_ctx** out) -> void
{
    require(out != nullptr, Status::null_argument, "Gram context output is null");
    *out = nullptr;
    validate_descriptor(descriptor);
    auto session = make_session(static_cast<cudaStream_t>(stream));
    auto& linalg = session->linalg();
    ctx_create_impl(descriptor, std::move(session), linalg, out);
}

auto ctx_create(const QnpepsGramDesc* descriptor, Linalg& linalg, qnpeps_gram_ctx** out) -> void
{
    require(out != nullptr, Status::null_argument, "Gram context output is null");
    *out = nullptr;
    validate_descriptor(descriptor);
    ctx_create_impl(descriptor, nullptr, linalg, out);
}

auto ctx_run(qnpeps_gram_ctx* context, const QnpepsGramArgs* args) -> void
{
    require(context != nullptr, Status::null_argument, "Gram run without a context");
    const std::lock_guard<std::mutex> lock{context->mutex};
    validate_args(*context, args);

    int caller{};
    check(cudaGetDevice(&caller));
    require(
        caller == context->device,
        Status::bad_config,
        "Gram run on device {}, the context lives on device {}",
        caller,
        context->device
    );

    BlasState blas_state{};
    context->linalg().blas_state(blas_state);
    const Defer restore_blas{[&] { context->linalg().restore_blas_state(blas_state); }};
    context->linalg().set_gram_state();

    const auto samples = static_cast<i64>(context->desc.n_samples);
    auto* gram = static_cast<cuFloatComplex*>(args->gram_out);
    const auto* rows = static_cast<const cuFloatComplex*>(args->o_rows);
    const auto* device_samples = static_cast<const std::uint8_t*>(args->samples);
    check(cudaMemsetAsync(
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
        if (destination.count < 1) continue;
        auto rows_a{rows + destination.base * context->compact};
        auto blockrow{gram + destination.base * samples};
        for (const VirtualShard& source : shards)
        {
            if (source.count < 1) continue;
            if (source.base == destination.base)
            {
                gram_slab::accumulate_block(
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
                check(cudaStreamSynchronize(context->stream()));
                continue;
            }
            for (i64 local{}; local < source.count; local += context->chunk_rows)
            {
                const i64 count{std::min(context->chunk_rows, source.count - local)};
                auto rows_b{rows + (source.base + local) * context->compact};
                gram_slab::accumulate_block(
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
                check(cudaStreamSynchronize(context->stream()));
            }
        }
        gram_slab::finalize_block(
            blockrow,
            static_cast<int>(samples),
            static_cast<int>(destination.count),
            static_cast<int>(destination.base),
            static_cast<int>(samples),
            context->stream()
        );
        check(cudaStreamSynchronize(context->stream()));
    }
}

auto ctx_footprint(const qnpeps_gram_ctx* context, QnpepsGramFootprint* out) -> void
{
    require(
        context != nullptr and out != nullptr,
        Status::null_argument,
        "Gram footprint query without a context or output"
    );
    require_struct_size(*out, "Gram footprint");
    *out = context->footprint;
}

auto ctx_destroy(qnpeps_gram_ctx* context) -> void
{
    if (not context) return;
    int caller{};
    const bool have_caller{cudaGetDevice(&caller) == cudaSuccess};
    const bool switched{
        have_caller and caller != context->device and cudaSetDevice(context->device) == cudaSuccess
    };
    unchecked(cudaStreamSynchronize(context->stream()));
    if (switched) unchecked(cudaSetDevice(caller));
    delete context;
}
}
