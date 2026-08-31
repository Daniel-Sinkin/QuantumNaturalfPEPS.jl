#include "dlenv/build/zipup_context.cuh"

namespace qnpeps::dlenv
{

static auto allocate_zipup_context(ZipupContext& context) -> void
{
    auto max_row_elements = 0_i64;
    for (auto row = 2; row <= context.config.lx; ++row)
        max_row_elements = std::max(max_row_elements, peps_row_elems(context.dims, row - 1, row));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&context.dl.peps_buf),
        static_cast<usize>(max_row_elements) * sizeof(cuFloatComplex)
    ));
    if (err_state() != QNPEPS_OK) return;

    const auto dim_bond = context.config.dim_bond;
    const auto chi_dl = std::min(context.maxdim, dim_bond * dim_bond);
    const auto density_route = context.config.dlenv_truncation_route == QNPEPS_TRUNCATION_DENSITY;
    const auto num_env_rows = static_cast<usize>(context.config.lx - 1);
    const auto num_cols = static_cast<usize>(context.config.ly);
    const auto scales_count = num_env_rows * num_cols;
    auto& context_arena = context.session->linalg().persistent_arena();
    BuildAllocation allocation{context.dl};
    allocation.carve(
        context.session->linalg(),
        context.dims,
        chi_dl,
        1,
        density_route,
        scales_count,
        context_arena
    );
    if (err_state() != QNPEPS_OK) return;

    init_dl_units(
        context.session->linalg(), context.dl.unit_environment, context.dl.initial_factor
    );
    context.dl.allocated = err_state() == QNPEPS_OK;
}

inline constexpr usize k_grouped_mps_axis_count{3};
inline constexpr usize k_grouped_mps_left{0};
inline constexpr usize k_grouped_mps_physical{1};
inline constexpr usize k_grouped_mps_right{2};

struct GroupedMpsDims
{
    int left{};
    int physical{};
    int right{};
};

[[nodiscard]] static auto read_grouped_mps_dims(const int32_t* dims, usize site) -> GroupedMpsDims
{
    const auto offset = site * k_grouped_mps_axis_count;
    return GroupedMpsDims{
        .left = dims[offset + k_grouped_mps_left],
        .physical = dims[offset + k_grouped_mps_physical],
        .right = dims[offset + k_grouped_mps_right],
    };
}

static auto write_grouped_mps_dims(int32_t* dims, usize site, const Shape& shape) -> void
{
    const auto offset = site * k_grouped_mps_axis_count;
    dims[offset + k_grouped_mps_left] = shape[k_grouped_mps_left];
    dims[offset + k_grouped_mps_physical] = shape[k_grouped_mps_physical];
    dims[offset + k_grouped_mps_right] = shape[k_grouped_mps_right];
}

[[nodiscard]] static auto append_product(u64& total, std::initializer_list<u64> factors) -> bool
{
    auto product = 1_u64;
    for (const auto factor : factors)
    {
        if (factor != 0 and product > std::numeric_limits<u64>::max() / factor) return false;
        product *= factor;
    }
    if (total > std::numeric_limits<u64>::max() - product) return false;
    total += product;
    return true;
}

[[nodiscard]] auto zipup_peps_row_bytes(const QnpepsConfig& config, int maxdim) -> i64
{
    if (maxdim < 1) return -1;
    const auto ly = static_cast<u64>(config.ly);
    const auto dim_bond = static_cast<u64>(config.dim_bond);
    if (dim_bond != 0 and dim_bond > std::numeric_limits<u64>::max() / dim_bond) return -1;
    const auto bond_pair = dim_bond * dim_bond;
    const auto capped_dim = std::min(static_cast<u64>(maxdim), bond_pair);
    auto elements = 0_u64;
    if (not append_product(elements, {ly, capped_dim, capped_dim, bond_pair})) return -1;
    if (elements > static_cast<u64>(std::numeric_limits<i64>::max()) / sizeof(cuFloatComplex))
        return -1;
    return static_cast<i64>(elements * sizeof(cuFloatComplex));
}

[[nodiscard]] static auto make_environment_views(
    const std::vector<DeviceTensor>& row_ket,
    const QnpepsZipupPepsRowArgs& args,
    std::vector<DeviceTensor>& environment
) -> qnpeps_status
{
    const auto has_dims = args.mps_dims != nullptr;
    const auto has_values = args.mps_values != nullptr;
    if (has_dims != has_values) return set_err(QNPEPS_ERR_NULL_ARG);
    if (not has_dims)
    {
        if (args.mps_bytes != 0) return set_err(QNPEPS_ERR_BAD_CONFIG);
        for (const auto& ket : row_ket)
            if (ket.dim[3] != 1) return set_err(QNPEPS_ERR_BAD_CONFIG);
        return QNPEPS_OK;
    }

    environment.resize(row_ket.size());
    const auto* values = reinterpret_cast<const cuFloatComplex*>(args.mps_values);
    auto previous_right = 1;
    auto value_elements = 0_u64;
    for (auto site = 0_uz; site < row_ket.size(); ++site)
    {
        const auto dims = read_grouped_mps_dims(args.mps_dims, site);
        if (dims.left < 1 or dims.physical < 1 or dims.right < 1)
            return set_err(QNPEPS_ERR_BAD_CONFIG);
        if (dims.left != previous_right) return set_err(QNPEPS_ERR_BAD_CONFIG);
        const auto vertical = row_ket[site].dim[3];
        if (static_cast<i64>(dims.physical) != static_cast<i64>(vertical) * vertical)
            return set_err(QNPEPS_ERR_BAD_CONFIG);
        const auto offset = value_elements;
        const auto product_appended = append_product(
            value_elements,
            {
                static_cast<u64>(dims.left),
                static_cast<u64>(dims.physical),
                static_cast<u64>(dims.right),
            }
        );
        if (not product_appended)
        {
            return set_err(QNPEPS_ERR_BAD_CONFIG);
        }
        environment[site] = DeviceTensor{
            {dims.left, vertical, vertical, dims.right},
            const_cast<cuFloatComplex*>(values + static_cast<usize>(offset))
        };
        previous_right = dims.right;
    }
    if (previous_right != 1) return set_err(QNPEPS_ERR_BAD_CONFIG);
    if (value_elements > std::numeric_limits<u64>::max() / sizeof(cuFloatComplex))
        return set_err(QNPEPS_ERR_BAD_CONFIG);
    if (args.mps_bytes != value_elements * sizeof(cuFloatComplex))
        return set_err(QNPEPS_ERR_BAD_CONFIG);
    return QNPEPS_OK;
}

[[nodiscard]] static auto copy_grouped_output(
    Linalg& linalg, const std::vector<DeviceTensor>& output, const QnpepsZipupPepsRowArgs& args
) -> qnpeps_status
{
    auto value_elements = 0_u64;
    for (auto site = 0_uz; site < output.size(); ++site)
    {
        if (output[site].dim.rank() != k_grouped_mps_axis_count)
            return set_err(QNPEPS_ERR_INTERNAL);
        write_grouped_mps_dims(args.output_dims, site, output[site].dim);
        const auto product_appended = append_product(
            value_elements,
            {
                static_cast<u64>(output[site].dim[0]),
                static_cast<u64>(output[site].dim[1]),
                static_cast<u64>(output[site].dim[2]),
            }
        );
        if (not product_appended)
        {
            return set_err(QNPEPS_ERR_INTERNAL);
        }
    }
    if (value_elements > std::numeric_limits<u64>::max() / sizeof(cuFloatComplex))
        return set_err(QNPEPS_ERR_INTERNAL);
    const auto value_bytes = value_elements * sizeof(cuFloatComplex);
    if (value_bytes > args.output_bytes) return set_err(QNPEPS_ERR_BAD_CONFIG);

    auto* destination = reinterpret_cast<cuFloatComplex*>(args.output_values);
    auto offset = 0_uz;
    for (const auto& site : output)
    {
        copy_device_async(linalg, destination + offset, site.d, site.num_elems());
        offset += site.num_elems();
    }
    return err_state();
}

auto create_zipup_context(const QnpepsConfig& config, int maxdim, cudaStream_t stream)
    -> qnpeps_zipup_ctx*
{
    auto session = make_session(stream, cudaStreamNonBlocking);
    if (not session) return nullptr;

    std::unique_ptr<ZipupContext> context{new (std::nothrow) ZipupContext{}};
    if (not context)
    {
        set_err(QNPEPS_ERR_OOM);
        return nullptr;
    }

    context->config = config;
    context->maxdim = maxdim;
    context->dims = Dims{config.lx, config.ly, config.dim_phys, config.dim_bond};
    context->session = std::move(session);
    allocate_zipup_context(*context);
    if (err_state() != QNPEPS_OK)
    {
        BuildAllocation allocation{context->dl};
        allocation.release();
        context.reset();
        return nullptr;
    }
    return reinterpret_cast<qnpeps_zipup_ctx*>(context.release());
}

auto destroy_zipup_context(qnpeps_zipup_ctx* opaque_context) -> void
{
    auto* context = zipup_context(opaque_context);
    if (not context) return;
    CUDA_NOCHECK(cudaStreamSynchronize(context->session->stream()));
    BuildAllocation allocation{context->dl};
    allocation.release();
    delete context;
}

auto begin_zipup_context(qnpeps_zipup_ctx& opaque_context) -> qnpeps_status
{
    auto& context = zipup_context(opaque_context);
    if (context.active) return set_err(QNPEPS_ERR_BAD_CONFIG);
    context.scale_count = 0;
    context.density_input_gauge = 0.0;
    zero_async(context.session->linalg(), context.dl.fail, 1);
    if (err_state() != QNPEPS_OK) return err_state();
    context.active = true;
    return QNPEPS_OK;
}

auto enqueue_peps_row(qnpeps_zipup_ctx& opaque_context, const QnpepsZipupPepsRowArgs& args)
    -> qnpeps_status
{
    auto& context = zipup_context(opaque_context);
    if (not context.active) return set_err(QNPEPS_ERR_BAD_CONFIG);
    if (args.row < 2 or args.row > context.config.lx) return set_err(QNPEPS_ERR_BAD_CONFIG);

    const auto row_elements = peps_row_elems(context.dims, args.row - 1, args.row);
    const auto expected_peps_bytes = static_cast<u64>(row_elements) * sizeof(cuFloatComplex);
    const auto required_output_bytes = zipup_peps_row_bytes(context.config, context.maxdim);
    const auto invalid_output_size =
        required_output_bytes < 0 or args.output_bytes < static_cast<u64>(required_output_bytes);
    if (args.peps_row_bytes != expected_peps_bytes or invalid_output_size)
    {
        return set_err(QNPEPS_ERR_BAD_CONFIG);
    }

    const auto num_cols = static_cast<usize>(context.config.ly);
    std::vector<DeviceTensor> row_ket{num_cols};
    auto source_offset = 0_i64;
    auto packed_offset = 0_i64;
    pack_peps_row(
        context.dims,
        args.row - 1,
        args.row,
        reinterpret_cast<const cuFloatComplex*>(args.peps_row),
        source_offset,
        context.dl.peps_buf,
        packed_offset,
        row_ket,
        context.session->stream()
    );
    if (source_offset != row_elements or packed_offset != row_elements)
        return set_err(QNPEPS_ERR_INTERNAL);

    std::vector<DeviceTensor> environment{};
    const auto environment_status = make_environment_views(row_ket, args, environment);
    if (environment_status != QNPEPS_OK) return environment_status;

    context.dl.known.rewind();
    context.dl.rolling_r.rewind();
    context.dl.scratch.rewind();
    if (context.config.dlenv_truncation_route == QNPEPS_TRUNCATION_DENSITY)
        context.dl.density.rewind();
    const Arenas arenas{context.dl.known, context.dl.rolling_r, context.dl.scratch};
    const auto scale_capacity = static_cast<usize>(context.config.lx - 1) * num_cols;
    if (context.scale_count + num_cols > scale_capacity) return set_err(QNPEPS_ERR_BAD_CONFIG);
    zipup::State state{
        .initial_factor = context.dl.initial_factor,
        .device_scales = context.dl.scales_all + context.scale_count,
        .fail_flag = context.dl.fail,
        .omegas = &context.dl.omegas,
        .rangefinder_rng = &context.dl.rangefinder_rng,
    };
    const auto density_route = context.config.dlenv_truncation_route == QNPEPS_TRUNCATION_DENSITY;
    std::vector<DeviceTensor> output{};
    if (density_route and not environment.empty())
    {
        EnvironmentRowBuilder row_builder{
            context.dl, context.session->linalg(), arenas, context.dims
        };
        auto density_output = row_builder.build_density_row(
            row_ket,
            environment,
            std::min(context.maxdim, context.config.dim_bond * context.config.dim_bond),
            context.config.dlenv_density_cutoff,
            static_cast<int>(context.scale_count / num_cols),
            context.density_input_gauge
        );
        output.resize(num_cols);
        for (auto site = 0_uz; site < num_cols; ++site)
        {
            output[site] = DeviceTensor{
                {density_output[site].dim[0],
                 density_output[site].dim[1] * density_output[site].dim[2],
                 density_output[site].dim[3]},
                density_output[site].d
            };
        }
    }
    else
    {
        const DeviceTensor unit_environment{{1, 1, 1, 1}, context.dl.unit_environment};
        output = zipup::fused_peps_row(
            state,
            context.session->linalg(),
            arenas,
            {
                .row_ket = &row_ket,
                .environment = environment.empty() ? nullptr : &environment,
                .unit_environment = unit_environment,
                .maxdim =
                    std::min(context.maxdim, context.config.dim_bond * context.config.dim_bond),
            }
        );
        if (density_route and environment.empty())
            zipup::accumulate_log_scales(state, num_cols, context.density_input_gauge);
    }
    if (err_state() != QNPEPS_OK) return err_state();
    const auto output_status = copy_grouped_output(context.session->linalg(), output, args);
    if (output_status != QNPEPS_OK) return output_status;
    context.scale_count += num_cols;
    return QNPEPS_OK;
}

auto finish_zipup_context(qnpeps_zipup_ctx& opaque_context, f64* scales, usize count)
    -> qnpeps_status
{
    auto& context = zipup_context(opaque_context);
    if (not context.active or count != context.scale_count) return set_err(QNPEPS_ERR_BAD_CONFIG);
    if (not scales) return set_err(QNPEPS_ERR_NULL_ARG);

    auto fail_host = 0;
    download_async(context.session->linalg(), scales, context.dl.scales_all, count);
    download_async(context.session->linalg(), &fail_host, context.dl.fail, 1);
    CUDA_CHECK(cudaStreamSynchronize(context.session->stream()));
    context.active = false;
    if (err_state() != QNPEPS_OK) return err_state();
    if (fail_host != 0) return set_err(QNPEPS_ERR_CUDA);
    for (auto index = 0_uz; index < count; ++index)
        if (not std::isfinite(scales[index])) return set_err(QNPEPS_ERR_INTERNAL);
    return QNPEPS_OK;
}
}
