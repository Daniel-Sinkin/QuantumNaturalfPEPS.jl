#include "core/error.cuh"
#include "core/predicates.cuh"
#include "e2e/layout.cuh"
#include "minsr/minsr.cuh"
#include "minsr/solve.cuh"

#include <climits>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>

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

[[nodiscard]] auto _descriptor_check(const QnpepsMinsrDesc* descriptor, Geometry& geometry) noexcept
    -> qnpeps_status
{
    if (not descriptor) return QNPEPS_ERR_NULL_ARG;
    if (descriptor->struct_size != sizeof(QnpepsMinsrDesc)) return QNPEPS_ERR_BAD_VERSION;
    if (descriptor->reserved != 0) return QNPEPS_ERR_BAD_CONFIG;
    const auto integer_limit = std::numeric_limits<int>::max();
    const auto valid_extent = qnpeps::in_range(descriptor->lx, k_minimum_extent, integer_limit)
                              and qnpeps::in_range(descriptor->ly, k_minimum_extent, integer_limit);
    const auto valid_dimensions =
        descriptor->dim_phys == k_physical_dimension and qnpeps::all_positive(descriptor->dim_bond);
    if (not valid_extent or not valid_dimensions) return QNPEPS_ERR_BAD_CONFIG;
    if (not qnpeps::in_range(descriptor->diagnostics, 0, 1)) return QNPEPS_ERR_BAD_CONFIG;
    const auto sample_limit = static_cast<qnpeps::i64>(INT_MAX);
    const auto valid_samples =
        qnpeps::in_range(descriptor->n_samples, k_minimum_sample_count, sample_limit);
    if (not valid_samples) return QNPEPS_ERR_BAD_CONFIG;
    if (not qnpeps::all_nonnegative(descriptor->host_tile_bytes)) return QNPEPS_ERR_BAD_CONFIG;

    geometry.config.struct_size = sizeof(QnpepsE2eConfig);
    geometry.config.lx = descriptor->lx;
    geometry.config.ly = descriptor->ly;
    geometry.config.dim_phys = descriptor->dim_phys;
    geometry.config.dim_bond = descriptor->dim_bond;
    geometry.sites = static_cast<qnpeps::i64>(descriptor->lx) * descriptor->ly;
    geometry.compact = qn_e2e::compact_count(geometry.config);
    if (not qnpeps::all_positive(geometry.compact)) return QNPEPS_ERR_BAD_CONFIG;
    const auto limit = std::numeric_limits<qnpeps::i64>::max();
    if (geometry.compact > limit / descriptor->dim_phys) return QNPEPS_ERR_BAD_CONFIG;
    geometry.dense = static_cast<qnpeps::i64>(descriptor->dim_phys) * geometry.compact;
    if (geometry.dense > limit / static_cast<qnpeps::i64>(sizeof(qnpeps::ComplexF32)))
        return QNPEPS_ERR_BAD_CONFIG;
    if (descriptor->n_samples > limit / geometry.compact) return QNPEPS_ERR_BAD_CONFIG;
    if (descriptor->n_samples > limit / descriptor->n_samples) return QNPEPS_ERR_BAD_CONFIG;
    return QNPEPS_OK;
}

[[nodiscard]] auto _args_check(
    const Geometry& geometry, const QnpepsMinsrDesc& descriptor, const QnpepsMinsrArgs* args
) noexcept -> qnpeps_status
{
    if (not args) return QNPEPS_ERR_NULL_ARG;
    if (args->struct_size != sizeof(QnpepsMinsrArgs)) return QNPEPS_ERR_BAD_VERSION;
    if (args->reserved != 0) return QNPEPS_ERR_BAD_CONFIG;
    const auto valid_inputs =
        args->samples and args->logpsi and args->e_loc and args->logq and args->gram;
    const auto valid_outputs =
        args->theta_dot_out and args->e_mean_out and args->e_var_out and args->ess_out;
    if (not valid_inputs or not valid_outputs) return QNPEPS_ERR_NULL_ARG;
    if (static_cast<bool>(args->o_rows_device) == static_cast<bool>(args->o_rows_host))
        return QNPEPS_ERR_NULL_ARG;

    const auto sample_count = static_cast<qnpeps::u64>(descriptor.n_samples);
    const auto site_count = static_cast<qnpeps::u64>(geometry.sites);
    const auto compact_count = static_cast<qnpeps::u64>(geometry.compact);
    const auto dense_count = static_cast<qnpeps::u64>(geometry.dense);
    const auto complex_bytes = static_cast<qnpeps::u64>(sizeof(qnpeps::ComplexF32));
    const auto real_bytes = static_cast<qnpeps::u64>(sizeof(qnpeps::f64));
    if (args->samples_bytes < sample_count * site_count) return QNPEPS_ERR_BAD_CONFIG;
    if (args->logpsi_bytes < k_complex_components * sample_count * real_bytes)
        return QNPEPS_ERR_BAD_CONFIG;
    if (args->e_loc_bytes < k_complex_components * sample_count * real_bytes)
        return QNPEPS_ERR_BAD_CONFIG;
    if (args->logq_bytes < sample_count * real_bytes) return QNPEPS_ERR_BAD_CONFIG;
    if (args->gram_bytes < sample_count * sample_count * complex_bytes)
        return QNPEPS_ERR_BAD_CONFIG;
    if (args->o_rows_bytes < sample_count * compact_count * complex_bytes)
        return QNPEPS_ERR_BAD_CONFIG;
    if (args->theta_dot_out_bytes < dense_count * complex_bytes) return QNPEPS_ERR_BAD_CONFIG;
    return QNPEPS_OK;
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

[[nodiscard]] auto descriptor_dense_count(const QnpepsMinsrDesc* descriptor) noexcept -> i64
{
    Geometry geometry{};
    return _descriptor_check(descriptor, geometry) == QNPEPS_OK ? geometry.dense : -1;
}

[[nodiscard]] auto descriptor_compact_count(const QnpepsMinsrDesc* descriptor) noexcept -> i64
{
    Geometry geometry{};
    return _descriptor_check(descriptor, geometry) == QNPEPS_OK ? geometry.compact : -1;
}

[[nodiscard]] auto descriptor_scratch_bytes(const QnpepsMinsrDesc* descriptor) -> i64
{
    Geometry geometry{};
    if (_descriptor_check(descriptor, geometry) != QNPEPS_OK) return -1;
    const auto bytes =
        qn_e2e_minsr_scratch(geometry.config, descriptor->n_samples, descriptor->host_tile_bytes);
    return qnpeps::err_state() == QNPEPS_OK ? static_cast<i64>(bytes) : -1;
}

auto ctx_create(const QnpepsMinsrDesc* descriptor, void* stream, qnpeps_minsr_ctx** out)
    -> qnpeps_status
{
    if (not out) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    *out = nullptr;
    Geometry geometry{};
    const auto descriptor_status = _descriptor_check(descriptor, geometry);
    if (descriptor_status != QNPEPS_OK) return qnpeps::set_err(descriptor_status);

    std::unique_ptr<qnpeps_minsr_ctx> context{new (std::nothrow) qnpeps_minsr_ctx};
    if (not context) return qnpeps::set_err(QNPEPS_ERR_OOM);
    context->descriptor = *descriptor;
    context->geometry = geometry;
    minsr_context_create(
        geometry.config,
        descriptor->n_samples,
        descriptor->host_tile_bytes,
        static_cast<cudaStream_t>(stream),
        &context->implementation
    );
    if (qnpeps::err_state() != QNPEPS_OK) return qnpeps::err_state();
    *out = context.release();
    return QNPEPS_OK;
}

auto ctx_run(qnpeps_minsr_ctx* context, const QnpepsMinsrArgs* args) -> qnpeps_status
{
    if (not context) return qnpeps::set_err(QNPEPS_ERR_NULL_ARG);
    const auto args_status = _args_check(context->geometry, context->descriptor, args);
    if (args_status != QNPEPS_OK) return qnpeps::set_err(args_status);
    const auto stream_mismatch = args->stream
                                 and static_cast<cudaStream_t>(args->stream)
                                         != minsr_context_stream(*context->implementation);
    if (stream_mismatch) return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);

    int device{};
    CUDA_CHECK(cudaGetDevice(&device));
    if (qnpeps::err_state() != QNPEPS_OK) return qnpeps::err_state();
    if (device != minsr_context_device(*context->implementation))
        return qnpeps::set_err(QNPEPS_ERR_BAD_CONFIG);
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
    return qnpeps::err_state();
}

auto ctx_destroy(qnpeps_minsr_ctx* context) noexcept -> void
{
    if (not context) return;
    minsr_context_destroy(context->implementation);
    delete context;
}

auto execute(const QnpepsMinsrDesc* descriptor, const QnpepsMinsrArgs* args) -> qnpeps_status
{
    Geometry geometry{};
    const auto descriptor_status = _descriptor_check(descriptor, geometry);
    if (descriptor_status != QNPEPS_OK) return qnpeps::set_err(descriptor_status);
    const auto args_status = _args_check(geometry, *descriptor, args);
    if (args_status != QNPEPS_OK) return qnpeps::set_err(args_status);

    qnpeps_minsr_ctx* context{};
    const auto create_status = ctx_create(descriptor, args->stream, &context);
    if (create_status != QNPEPS_OK) return create_status;
    const auto run_status = ctx_run(context, args);
    ctx_destroy(context);
    return run_status;
}

}
