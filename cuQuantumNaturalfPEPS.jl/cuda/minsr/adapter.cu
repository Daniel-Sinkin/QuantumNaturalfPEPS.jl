#include "capi/internal/format.cuh"
#include "core/defer.cuh"
#include "core/error.hpp"
#include "core/predicates.cuh"
#include "e2e/layout.cuh"
#include "minsr/minsr.cuh"
#include "minsr/solve.cuh"

#include <climits>
#include <cstdint>
#include <limits>
#include <memory>

namespace
{
constexpr int k_physical_dimension{2};
constexpr int k_minimum_extent{2};
constexpr qnpeps::i64 k_minimum_sample_count{2};
constexpr qnpeps::u64 k_complex_components{2};

struct Geometry
{
    QnpepsE2eConfig config{};
    qnpeps::i64 sites{};
    qnpeps::i64 compact{};
    qnpeps::i64 dense{};
};

auto validate_descriptor(const QnpepsMinsrDesc* descriptor) -> Geometry
{
    using qnpeps::Status;
    qnpeps::require(descriptor != nullptr, Status::null_argument, "MinSR descriptor is null");
    qnpeps::require_struct_size(*descriptor, "MinSR descriptor");
    qnpeps::require(
        descriptor->reserved == 0,
        Status::bad_config,
        "MinSR descriptor reserved field is {}",
        descriptor->reserved
    );
    const auto integer_limit = std::numeric_limits<int>::max();
    const auto valid_extent = qnpeps::in_range(descriptor->lx, k_minimum_extent, integer_limit)
                              and qnpeps::in_range(descriptor->ly, k_minimum_extent, integer_limit);
    const auto valid_dimensions =
        descriptor->dim_phys == k_physical_dimension and qnpeps::all_positive(descriptor->dim_bond);
    qnpeps::require(
        valid_extent and valid_dimensions,
        Status::bad_config,
        "MinSR descriptor rejects {}",
        *descriptor
    );
    qnpeps::require(
        qnpeps::in_range(descriptor->diagnostics, 0, 1),
        Status::bad_config,
        "MinSR diagnostics flag is neither 0 nor 1, {}",
        *descriptor
    );
    const auto sample_limit = static_cast<qnpeps::i64>(INT_MAX);
    qnpeps::require(
        qnpeps::in_range(descriptor->n_samples, k_minimum_sample_count, sample_limit),
        Status::bad_config,
        "MinSR sample count outside {} to {}, {}",
        k_minimum_sample_count,
        sample_limit,
        *descriptor
    );
    qnpeps::require(
        qnpeps::all_nonnegative(descriptor->host_tile_bytes),
        Status::bad_config,
        "MinSR host tile bytes are negative, {}",
        *descriptor
    );

    Geometry geometry{};
    geometry.config.struct_size = sizeof(QnpepsE2eConfig);
    geometry.config.lx = descriptor->lx;
    geometry.config.ly = descriptor->ly;
    geometry.config.dim_phys = descriptor->dim_phys;
    geometry.config.dim_bond = descriptor->dim_bond;
    geometry.sites = static_cast<qnpeps::i64>(descriptor->lx) * descriptor->ly;
    geometry.compact = qn_e2e::compact_count(geometry.config);
    const auto limit = std::numeric_limits<qnpeps::i64>::max();
    const auto counts_fit =
        qnpeps::all_positive(geometry.compact) and geometry.compact <= limit / descriptor->dim_phys;
    qnpeps::require(
        counts_fit, Status::bad_config, "MinSR compact count {} out of range", geometry.compact
    );
    geometry.dense = static_cast<qnpeps::i64>(descriptor->dim_phys) * geometry.compact;
    const auto sizes_fit =
        geometry.dense <= limit / static_cast<qnpeps::i64>(sizeof(qnpeps::ComplexF32))
        and descriptor->n_samples <= limit / geometry.compact
        and descriptor->n_samples <= limit / descriptor->n_samples;
    qnpeps::require(
        sizes_fit,
        Status::bad_config,
        "MinSR sizes for {} samples and {} parameters overflow",
        descriptor->n_samples,
        geometry.dense
    );
    return geometry;
}

auto validate_args(
    const Geometry& geometry, const QnpepsMinsrDesc& descriptor, const QnpepsMinsrArgs* args
) -> void
{
    using qnpeps::Status;
    qnpeps::require(args != nullptr, Status::null_argument, "MinSR arguments are null");
    qnpeps::require_struct_size(*args, "MinSR arguments");
    qnpeps::require(
        args->reserved == 0,
        Status::bad_config,
        "MinSR arguments reserved field is {}",
        args->reserved
    );
    const auto valid_inputs =
        args->samples and args->logpsi and args->e_loc and args->logq and args->gram;
    const auto valid_outputs =
        args->theta_dot_out and args->e_mean_out and args->e_var_out and args->ess_out;
    qnpeps::require(
        valid_inputs and valid_outputs, Status::null_argument, "MinSR arguments miss a buffer"
    );
    qnpeps::require(
        static_cast<bool>(args->o_rows_device) != static_cast<bool>(args->o_rows_host),
        Status::null_argument,
        "MinSR needs exactly one of device rows and host rows"
    );

    const auto sample_count = static_cast<qnpeps::u64>(descriptor.n_samples);
    const auto site_count = static_cast<qnpeps::u64>(geometry.sites);
    const auto compact_count = static_cast<qnpeps::u64>(geometry.compact);
    const auto dense_count = static_cast<qnpeps::u64>(geometry.dense);
    const auto complex_bytes = static_cast<qnpeps::u64>(sizeof(qnpeps::ComplexF32));
    const auto real_bytes = static_cast<qnpeps::u64>(sizeof(qnpeps::f64));
    const auto require_bytes = [](qnpeps::u64 actual, qnpeps::u64 needed, const char* buffer)
    {
        qnpeps::require(
            actual >= needed,
            Status::bad_config,
            "MinSR {} buffer holds {} bytes, needs {}",
            buffer,
            actual,
            needed
        );
    };
    require_bytes(args->samples_bytes, sample_count * site_count, "samples");
    require_bytes(args->logpsi_bytes, k_complex_components * sample_count * real_bytes, "logpsi");
    require_bytes(args->e_loc_bytes, k_complex_components * sample_count * real_bytes, "e_loc");
    require_bytes(args->logq_bytes, sample_count * real_bytes, "logq");
    require_bytes(args->gram_bytes, sample_count * sample_count * complex_bytes, "gram");
    require_bytes(args->o_rows_bytes, sample_count * compact_count * complex_bytes, "o_rows");
    require_bytes(args->theta_dot_out_bytes, dense_count * complex_bytes, "theta_dot_out");
}

}

struct qnpeps_minsr_ctx
{
    QnpepsMinsrDesc descriptor{};
    Geometry geometry{};
    qnpeps_e2e_minsr_ctx* implementation{};
};

namespace qnpeps::minsr
{

[[nodiscard]] auto descriptor_dense_count(const QnpepsMinsrDesc* descriptor) -> i64
{
    return validate_descriptor(descriptor).dense;
}

[[nodiscard]] auto descriptor_compact_count(const QnpepsMinsrDesc* descriptor) -> i64
{
    return validate_descriptor(descriptor).compact;
}

[[nodiscard]] auto descriptor_scratch_bytes(const QnpepsMinsrDesc* descriptor) -> i64
{
    const auto geometry = validate_descriptor(descriptor);
    const auto bytes =
        qn_e2e_minsr_scratch(geometry.config, descriptor->n_samples, descriptor->host_tile_bytes);
    return static_cast<i64>(bytes);
}

auto ctx_create(const QnpepsMinsrDesc* descriptor, void* stream, qnpeps_minsr_ctx** out) -> void
{
    require(out != nullptr, Status::null_argument, "MinSR context output is null");
    *out = nullptr;
    const auto geometry = validate_descriptor(descriptor);

    auto context = std::make_unique<qnpeps_minsr_ctx>();
    context->descriptor = *descriptor;
    context->geometry = geometry;
    minsr_context_create(
        geometry.config,
        descriptor->n_samples,
        descriptor->host_tile_bytes,
        static_cast<cudaStream_t>(stream),
        &context->implementation
    );
    *out = context.release();
}

auto ctx_run(qnpeps_minsr_ctx* context, const QnpepsMinsrArgs* args) -> void
{
    require(context != nullptr, Status::null_argument, "MinSR run without a context");
    validate_args(context->geometry, context->descriptor, args);
    const auto stream_mismatch = args->stream
                                 and static_cast<cudaStream_t>(args->stream)
                                         != minsr_context_stream(*context->implementation);
    require(
        not stream_mismatch, Status::bad_config, "MinSR run stream differs from the context stream"
    );

    int device{};
    check(cudaGetDevice(&device));
    const auto context_device = minsr_context_device(*context->implementation);
    require(
        device == context_device,
        Status::bad_config,
        "MinSR run on device {}, the context lives on device {}",
        device,
        context_device
    );
    minsr_context_run(
        *context->implementation,
        static_cast<const u8*>(args->samples),
        args->logpsi,
        args->e_loc,
        args->logq,
        static_cast<const cf*>(args->gram),
        static_cast<const cf*>(args->o_rows_device),
        static_cast<const cf*>(args->o_rows_host),
        args->relative_cut,
        args->absolute_cut,
        static_cast<cf*>(args->theta_dot_out),
        args->e_mean_out,
        args->e_var_out,
        args->ess_out
    );
}

auto ctx_destroy(qnpeps_minsr_ctx* context) -> void
{
    if (not context) return;
    minsr_context_destroy(context->implementation);
    delete context;
}

auto execute(const QnpepsMinsrDesc* descriptor, const QnpepsMinsrArgs* args) -> void
{
    const auto geometry = validate_descriptor(descriptor);
    validate_args(geometry, *descriptor, args);

    qnpeps_minsr_ctx* context{};
    ctx_create(descriptor, args->stream, &context);
    const Defer destroy_context{[&] { ctx_destroy(context); }};
    ctx_run(context, args);
}

}
