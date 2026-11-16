#include "dlenv/build/zipup_context.cuh"

namespace qnpeps::dlenv
{

static auto allocate_zipup_context(ZipupContext& context) -> void
{
    auto max_row_elements = 0_i64;
    for (auto row = 2; row <= context.config.lx; ++row)
        max_row_elements = std::max(max_row_elements, peps_row_elems(context.dims, row - 1, row));
    check(cudaMalloc(
        reinterpret_cast<void**>(&context.dl.peps_buf),
        static_cast<usize>(max_row_elements) * sizeof(cuFloatComplex)
    ));

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

    init_dl_units(
        context.session->linalg(), context.dl.unit_environment, context.dl.initial_factor
    );
    context.dl.allocated = true;
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

static auto make_environment_views(
    const std::vector<DeviceTensor>& row_ket,
    const QnpepsZipupPepsRowArgs& args,
    std::vector<DeviceTensor>& environment
) -> void
{
    const auto has_dims = args.mps_dims != nullptr;
    const auto has_values = args.mps_values != nullptr;
    require(
        has_dims == has_values,
        Status::null_argument,
        "zipup row environment passes dimensions without values or values without dimensions"
    );
    if (not has_dims)
    {
        require(
            args.mps_bytes == 0,
            Status::bad_config,
            "zipup row without an environment passes {} environment bytes",
            args.mps_bytes
        );
        for (const auto& ket : row_ket)
        {
            require(
                ket.dim[3] == 1,
                Status::bad_config,
                "zipup row without an environment has a ket with lower bond {}",
                ket.dim[3]
            );
        }
        return;
    }

    environment.resize(row_ket.size());
    const auto* values = reinterpret_cast<const cuFloatComplex*>(args.mps_values);
    auto previous_right = 1;
    auto value_elements = 0_u64;
    for (auto site = 0_uz; site < row_ket.size(); ++site)
    {
        const auto dims = read_grouped_mps_dims(args.mps_dims, site);
        const auto positive = dims.left >= 1 and dims.physical >= 1 and dims.right >= 1;
        require(
            positive,
            Status::bad_config,
            "zipup environment site {} has extents {}x{}x{}",
            site,
            dims.left,
            dims.physical,
            dims.right
        );
        require(
            dims.left == previous_right,
            Status::bad_config,
            "zipup environment site {} left bond {} does not match the previous right bond {}",
            site,
            dims.left,
            previous_right
        );
        const auto vertical = row_ket[site].dim[3];
        require(
            static_cast<i64>(dims.physical) == static_cast<i64>(vertical) * vertical,
            Status::bad_config,
            "zipup environment site {} physical extent {} is not the squared ket bond {}",
            site,
            dims.physical,
            vertical
        );
        const auto offset = value_elements;
        const auto product_appended = append_product(
            value_elements,
            {
                static_cast<u64>(dims.left),
                static_cast<u64>(dims.physical),
                static_cast<u64>(dims.right),
            }
        );
        require(
            product_appended,
            Status::bad_config,
            "zipup environment element count overflows at site {}",
            site
        );
        environment[site] = DeviceTensor{
            {dims.left, vertical, vertical, dims.right},
            const_cast<cuFloatComplex*>(values + static_cast<usize>(offset))
        };
        previous_right = dims.right;
    }
    require(
        previous_right == 1,
        Status::bad_config,
        "zipup environment ends in right bond {}",
        previous_right
    );
    require(
        value_elements <= std::numeric_limits<u64>::max() / sizeof(cuFloatComplex),
        Status::bad_config,
        "zipup environment of {} elements overflows the byte range",
        value_elements
    );
    require(
        args.mps_bytes == value_elements * sizeof(cuFloatComplex),
        Status::bad_config,
        "zipup environment holds {} bytes, its dimensions need {}",
        args.mps_bytes,
        value_elements * sizeof(cuFloatComplex)
    );
}

static auto copy_grouped_output(
    Linalg& linalg, const std::vector<DeviceTensor>& output, const QnpepsZipupPepsRowArgs& args
) -> void
{
    auto value_elements = 0_u64;
    for (auto site = 0_uz; site < output.size(); ++site)
    {
        require(
            output[site].dim.rank() == k_grouped_mps_axis_count,
            "zipup output site {} has rank {}",
            site,
            output[site].dim.rank()
        );
        write_grouped_mps_dims(args.output_dims, site, output[site].dim);
        const auto product_appended = append_product(
            value_elements,
            {
                static_cast<u64>(output[site].dim[0]),
                static_cast<u64>(output[site].dim[1]),
                static_cast<u64>(output[site].dim[2]),
            }
        );
        require(product_appended, "zipup output element count overflows at site {}", site);
    }
    require(
        value_elements <= std::numeric_limits<u64>::max() / sizeof(cuFloatComplex),
        "zipup output of {} elements overflows the byte range",
        value_elements
    );
    const auto value_bytes = value_elements * sizeof(cuFloatComplex);
    require(
        value_bytes <= args.output_bytes,
        Status::bad_config,
        "zipup output needs {} bytes, the buffer holds {}",
        value_bytes,
        args.output_bytes
    );

    auto* destination = reinterpret_cast<cuFloatComplex*>(args.output_values);
    auto offset = 0_uz;
    for (const auto& site : output)
    {
        copy_device_async(linalg, destination + offset, site.d, site.num_elems());
        offset += site.num_elems();
    }
}

auto create_zipup_context(const QnpepsConfig& config, int maxdim, cudaStream_t stream)
    -> qnpeps_zipup_ctx*
{
    auto session = make_session(stream, cudaStreamNonBlocking);
    auto context = std::make_unique<ZipupContext>();
    context->config = config;
    context->maxdim = maxdim;
    context->dims = Dims{config.lx, config.ly, config.dim_phys, config.dim_bond};
    context->session = std::move(session);
    auto created = false;
    const Defer release_after_failure{[&]
                                      {
                                          if (created) return;
                                          BuildAllocation allocation{context->dl};
                                          allocation.release();
                                      }};
    allocate_zipup_context(*context);
    created = true;
    return reinterpret_cast<qnpeps_zipup_ctx*>(context.release());
}

auto destroy_zipup_context(qnpeps_zipup_ctx* opaque_context) -> void
{
    auto* context = zipup_context(opaque_context);
    if (not context) return;
    unchecked(cudaStreamSynchronize(context->session->stream()));
    BuildAllocation allocation{context->dl};
    allocation.release();
    delete context;
}

auto begin_zipup_context(qnpeps_zipup_ctx& opaque_context) -> void
{
    auto& context = zipup_context(opaque_context);
    require(not context.active, Status::bad_config, "zipup context begins while a sweep is active");
    context.scale_count = 0;
    context.density_input_gauge = 0.0;
    zero_async(context.session->linalg(), context.dl.fail, 1);
    context.active = true;
}

auto enqueue_peps_row(qnpeps_zipup_ctx& opaque_context, const QnpepsZipupPepsRowArgs& args) -> void
{
    auto& context = zipup_context(opaque_context);
    require(context.active, Status::bad_config, "zipup row enqueued before begin");
    require(
        args.row >= 2 and args.row <= context.config.lx,
        Status::bad_config,
        "zipup row {} outside 2 to {}",
        args.row,
        context.config.lx
    );

    const auto row_elements = peps_row_elems(context.dims, args.row - 1, args.row);
    const auto expected_peps_bytes = static_cast<u64>(row_elements) * sizeof(cuFloatComplex);
    const auto required_output_bytes = zipup_peps_row_bytes(context.config, context.maxdim);
    const auto invalid_output_size =
        required_output_bytes < 0 or args.output_bytes < static_cast<u64>(required_output_bytes);
    require(
        args.peps_row_bytes == expected_peps_bytes,
        Status::bad_config,
        "zipup row {} passes {} PEPS bytes, expected {}",
        args.row,
        args.peps_row_bytes,
        expected_peps_bytes
    );
    require(
        not invalid_output_size,
        Status::bad_config,
        "zipup row output holds {} bytes, needs {}",
        args.output_bytes,
        required_output_bytes
    );

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
    const auto packed_whole_row = source_offset == row_elements and packed_offset == row_elements;
    require(
        packed_whole_row,
        "zipup row {} packed {} of {} elements",
        args.row,
        packed_offset,
        row_elements
    );

    std::vector<DeviceTensor> environment{};
    make_environment_views(row_ket, args, environment);

    context.dl.known.rewind();
    context.dl.rolling_r.rewind();
    context.dl.scratch.rewind();
    if (context.config.dlenv_truncation_route == QNPEPS_TRUNCATION_DENSITY)
        context.dl.density.rewind();
    const Arenas arenas{context.dl.known, context.dl.rolling_r, context.dl.scratch};
    const auto scale_capacity = static_cast<usize>(context.config.lx - 1) * num_cols;
    require(
        context.scale_count + num_cols <= scale_capacity,
        Status::bad_config,
        "zipup sweep holds {} scales, row {} needs {} more of capacity {}",
        context.scale_count,
        args.row,
        num_cols,
        scale_capacity
    );
    zipup::State state{
        .initial_factor = context.dl.initial_factor,
        .device_scales = context.dl.scales_all + context.scale_count,
        .fail_flag = context.dl.fail,
        .power_input_exponent = context.dl.fail + 1,
        .omegas = &context.dl.omegas,
        .rangefinder_rng = &context.dl.rangefinder_rng,
        .svd_route = context.config.dlenv_truncation_route == QNPEPS_TRUNCATION_SVD,
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
    copy_grouped_output(context.session->linalg(), output, args);
    context.scale_count += num_cols;
}

auto finish_zipup_context(qnpeps_zipup_ctx& opaque_context, f64* scales, usize count) -> void
{
    auto& context = zipup_context(opaque_context);
    require(context.active, Status::bad_config, "zipup context finishes without an active sweep");
    const Defer end_sweep{[&] { context.active = false; }};
    require(
        count == context.scale_count,
        Status::bad_config,
        "zipup finish asks for {} scales, the sweep holds {}",
        count,
        context.scale_count
    );
    require(scales != nullptr, Status::null_argument, "zipup finish has no scale output");

    auto fail_host = 0;
    download_async(context.session->linalg(), scales, context.dl.scales_all, count);
    download_async(context.session->linalg(), &fail_host, context.dl.fail, 1);
    check(cudaStreamSynchronize(context.session->stream()));
    zipup::raise_rangefinder_failure(fail_host);
    for (auto index = 0_uz; index < count; ++index)
    {
        require(std::isfinite(scales[index]), "zipup scale {} of {} is not finite", index, count);
    }
}
}
