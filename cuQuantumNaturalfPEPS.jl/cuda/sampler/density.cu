#include "core/complex.cuh"
#include "core/cuda_utils.cuh"
#include "densitymatrix/types.cuh"
#include "linalg/transfer.cuh"
#include "sampler/density.cuh"

#include <algorithm>
#include <span>
#include <vector>

namespace qnpeps::sampler
{
namespace
{
struct ProductContext
{
    Sampler* sampler{};
    const SamplerConfig* config{};
    int row{};
    bool projected{};
};

struct ProductArgs
{
    const cuDoubleComplex* state{};
    const cuFloatComplex* peps{};
    const int* chosen_spin{};
    i64 state_left{};
    i64 state_right{};
    i64 left{};
    i64 up{};
    i64 dim_phys{};
    i64 down{};
    i64 right{};
    i64 output_state{};
    bool projected{};
    cuDoubleComplex* product{};
};

struct PromoteArgs
{
    const cuFloatComplex* input{};
    usize count{};
    cuDoubleComplex* output{};
};

struct PackArgs
{
    const cuDoubleComplex* input{};
    i64 physical{};
    i64 right{};
    i64 left{};
    cuFloatComplex* output{};
};

__global__ auto cu_product(ProductArgs args) -> void
{
    const auto combined_left = args.state_left * args.left;
    const auto combined_right = args.state_right * args.right;
    const auto count = combined_left * combined_right;
    for (auto lane = global_lane(); lane < count; lane += grid_stride())
    {
        const auto combined_left_index = lane % combined_left;
        const auto combined_right_index = lane / combined_left;
        const auto state_left = combined_left_index % args.state_left;
        const auto operator_left = combined_left_index / args.state_left;
        const auto state_right = combined_right_index % args.state_right;
        const auto operator_right = combined_right_index / args.state_right;
        const auto physical = args.projected ? static_cast<i64>(*args.chosen_spin)
                                             : args.output_state % args.dim_phys;
        const auto down = args.projected ? args.output_state : args.output_state / args.dim_phys;
        auto value = make_cuDoubleComplex(0.0, 0.0);
        for (auto up = 0_i64; up < args.up; ++up)
        {
            const auto state_index = state_left + args.state_left * (up + args.up * state_right);
            const auto peps_index =
                operator_left
                + args.left
                      * (up
                         + args.up
                               * (physical + args.dim_phys * (down + args.down * operator_right)));
            const auto peps = make_cuDoubleComplex(
                static_cast<f64>(args.peps[peps_index].x), static_cast<f64>(args.peps[peps_index].y)
            );
            value = cuCadd(value, cuCmul(args.state[state_index], peps));
        }
        args.product[lane] = value;
    }
}

__global__ auto cu_promote(PromoteArgs args) -> void
{
    for (auto lane = global_lane(); lane < static_cast<i64>(args.count); lane += grid_stride())
    {
        args.output[lane] = make_cuDoubleComplex(
            static_cast<f64>(args.input[lane].x), static_cast<f64>(args.input[lane].y)
        );
    }
}

__global__ auto cu_pack(PackArgs args) -> void
{
    const auto count = args.left * args.physical * args.right;
    for (auto lane = global_lane(); lane < count; lane += grid_stride())
    {
        const auto left = lane % args.left;
        const auto tail = lane / args.left;
        const auto physical = tail % args.physical;
        const auto right = tail / args.physical;
        const auto value = args.input[physical + args.physical * (right + args.right * left)];
        args.output[lane] = to_cf<cuFloatComplex>(value);
    }
}

__global__ auto cu_add_log(const f64* input, f64* output) -> void
{
    if (global_lane() == 0) *output += *input;
}

auto shift(CuSpanCF32& array, int lane) -> void
{
    if (array.p) array.p += static_cast<i64>(lane) * array.stride;
}

auto shift(cuFloatComplex**& pointers, int lane) -> void
{
    if (pointers) pointers += lane;
}

auto shift_sampler(Sampler& sampler, int lane) -> void
{
    shift(sampler.env_above()[0], lane);
    shift(sampler.env_above()[1], lane);
    shift(sampler.ket(), lane);
    shift(sampler.env_unsampled(), lane);
    shift(sampler.sigma(), lane);
    shift(sampler.sigma_full(), lane);
    shift(sampler.sigma_full_scratch(), lane);
    shift(sampler.rho(), lane);
    shift(sampler.rfactor(), lane);
    shift(sampler.tmp_a(), lane);
    shift(sampler.tmp_b(), lane);
    shift(sampler.reduce_input(), lane);
    shift(sampler.sketch(), lane);
    shift(sampler.projection(), lane);
    shift(sampler.rfactor_next(), lane);
    shift(sampler.gram(), lane);
    shift(sampler.gram_ptrs(), lane);
    shift(sampler.sketch_ptrs(), lane);
    shift(sampler.tmp_a_ptrs(), lane);
    shift(sampler.tmp_b_ptrs(), lane);
    shift(sampler.dl_unit_ptrs(), lane);
    for (auto*& pointers : sampler.envu_ptrs())
        shift(pointers, lane);
    for (auto*& pointers : sampler.ket_row0_ptrs())
        shift(pointers, lane);
    for (auto& row : sampler.mpo_ptrs())
    {
        for (auto*& pointers : row)
            shift(pointers, lane);
    }
    for (auto& row : sampler.dlenv_env_ptrs())
    {
        for (auto*& pointers : row)
            shift(pointers, lane);
    }
    for (auto& row : sampler.dlenv_sigma_ptrs())
    {
        for (auto*& pointers : row)
            shift(pointers, lane);
    }
    if (sampler.info()) sampler.info() += lane;
    if (sampler.drawn_spin()) sampler.drawn_spin() += lane;
    if (sampler.row_spins()) sampler.row_spins() += lane;
    if (sampler.logpc()) sampler.logpc() += lane;
    if (sampler.lognorm()) sampler.lognorm() += lane;
    if (sampler.samples()) sampler.samples() += static_cast<i64>(lane) * sampler.cfg().num_sites();
}

auto product(
    Linalg& linalg,
    const densitymatrix::Site& site,
    i64 output_state,
    const cuDoubleComplex* state,
    const cuDoubleComplex*,
    cuDoubleComplex* output,
    const void* context_value
) -> void
{
    const auto& context = *static_cast<const ProductContext*>(context_value);
    auto& sampler = *context.sampler;
    const auto site_index = static_cast<usize>(site.site);
    const auto& shape = sampler.peps_shapes()[static_cast<usize>(context.row)][site_index];
    const auto combined_left = site.state_left * site.operator_left;
    const auto combined_right = site.state_right * site.operator_right;
    const auto* chosen =
        context.projected
            ? sampler.row_spins() + static_cast<i64>(site.site) * context.config->row_spin_stride
            : nullptr;
    const auto product_args = ProductArgs{
        .state = state + site.state_offset,
        .peps = sampler.mpo()[static_cast<usize>(context.row)][site_index],
        .chosen_spin = chosen,
        .state_left = site.state_left,
        .state_right = site.state_right,
        .left = shape[0],
        .up = shape[3],
        .dim_phys = shape[4],
        .down = shape[1],
        .right = shape[2],
        .output_state = output_state,
        .projected = context.projected,
        .product = output,
    };
    cu_product<<<
        grid_blocks_capped(combined_left * combined_right),
        k_threads_per_block,
        0,
        linalg.stream()>>>(product_args);
    CUDA_CHECK(cudaGetLastError());
}

auto apply_row(
    qnpeps_ctx& ctx,
    const SamplerConfig& config,
    int row,
    int input_environment_index,
    std::span<const int> input_bonds,
    std::vector<int>& output_bonds,
    bool projected,
    int upper_bond,
    f64 cutoff,
    cuFloatComplex* output,
    i64 output_site_stride,
    bool normalize
) -> bool
{
    auto& sampler = ctx.sampler.samp;
    auto& storage = ctx.sampler.density;
    const auto sites = static_cast<usize>(config.ly);
    std::vector<densitymatrix::Site> plans{};
    plans.reserve(sites);
    auto state_offset = 0_uz;
    const auto& input = sampler.env_above()[static_cast<usize>(input_environment_index)];
    for (auto site_index = 0_uz; site_index < sites; ++site_index)
    {
        const auto& shape = sampler.peps_shapes()[static_cast<usize>(row)][site_index];
        const auto left = input_bonds[site_index];
        const auto right = input_bonds[site_index + 1];
        const auto input_count = static_cast<usize>(left * shape[3] * right);
        const auto promote_args = PromoteArgs{
            .input = input.p + static_cast<i64>(site_index) * sampler.max_env_above_site(),
            .count = input_count,
            .output = storage.state_values + state_offset,
        };
        cu_promote<<<
            grid_blocks_capped(static_cast<i64>(input_count)),
            k_threads_per_block,
            0,
            ctx.linalg().stream()>>>(promote_args);
        plans.push_back(
            {.site = site_index,
             .num_sites = sites,
             .state_left = left,
             .physical_input = shape[3],
             .state_right = right,
             .operator_left = shape[0],
             .physical_output = projected ? shape[1] : shape[4] * shape[1],
             .operator_right = shape[2],
             .state_offset = state_offset,
             .operator_offset = 0,
             .output_offset = site_index * storage.result_site_stride}
        );
        state_offset += input_count;
    }
    CUDA_CHECK(cudaGetLastError());
    if (state_offset > storage.state_value_count or err_state() != QNPEPS_OK) return false;
    const ProductContext context{&sampler, &config, row, projected};
    densitymatrix::apply(
        ctx.linalg(),
        {.settings =
             {.struct_size = static_cast<u32>(sizeof(QnpepsDensitySettings)),
              .precision = 1_u32,
              .mindim = 1_u32,
              .reserved = 0_u32,
              .relative_cutoff = cutoff},
         .upper_bond = upper_bond,
         .normalize = normalize,
         .input_gauge = 0.0,
         .sites = plans,
         .workspace = storage.workspace,
         .state_values = storage.state_values,
         .state_value_count = state_offset,
         .operator_values = nullptr,
         .operator_value_count = 0,
         .result_dimensions = storage.result_dimensions,
         .result_dimension_count = 3 * sites,
         .result_values = storage.result_values,
         .result_value_count = storage.result_value_count,
         .normalization_log = storage.normalization_log,
         .output_gauge = storage.output_gauge,
         .product = product,
         .product_context = &context}
    );
    if (err_state() != QNPEPS_OK) return false;
    download_async(
        ctx.linalg(), storage.host_dimensions.data(), storage.result_dimensions, 3 * sites
    );
    CUDA_CHECK(cudaStreamSynchronize(ctx.linalg().stream()));
    if (err_state() != QNPEPS_OK) return false;
    output_bonds.assign(sites + 1, 1);
    for (auto site_index = 0_uz; site_index < sites; ++site_index)
    {
        const auto dimension_offset = 3 * site_index;
        const auto physical = static_cast<i64>(storage.host_dimensions[dimension_offset]);
        const auto right = static_cast<i64>(storage.host_dimensions[dimension_offset + 1]);
        const auto left = static_cast<i64>(storage.host_dimensions[dimension_offset + 2]);
        if (left < 1 or right < 1 or physical < 1) return set_err(QNPEPS_ERR_INTERNAL), false;
        output_bonds[site_index] = static_cast<int>(left);
        output_bonds[site_index + 1] = static_cast<int>(right);
        const auto pack_args = PackArgs{
            .input = storage.result_values + site_index * storage.result_site_stride,
            .physical = physical,
            .right = right,
            .left = left,
            .output = output + static_cast<i64>(site_index) * output_site_stride,
        };
        cu_pack<<<
            grid_blocks_capped(left * physical * right),
            k_threads_per_block,
            0,
            ctx.linalg().stream()>>>(pack_args);
    }
    if (normalize)
    {
        cu_add_log<<<1, 1, 0, ctx.linalg().stream()>>>(
            storage.normalization_log, sampler.lognorm()
        );
    }
    CUDA_CHECK(cudaGetLastError());
    return err_state() == QNPEPS_OK;
}
}

DensityLane::DensityLane(Sampler& sampler, int lane) : sampler_(sampler), lane_(lane)
{
    shift_sampler(sampler_, lane_);
}

DensityLane::~DensityLane()
{
    shift_sampler(sampler_, -lane_);
}

auto build_density_state_row(
    qnpeps_ctx& ctx,
    const SamplerConfig& config,
    int row,
    int environment_index,
    std::span<const int> input_bonds,
    std::vector<int>& output_bonds
) -> bool
{
    return apply_row(
        ctx,
        config,
        row,
        environment_index,
        input_bonds,
        output_bonds,
        false,
        config.chi_s,
        config.state_density_cutoff,
        ctx.sampler.samp.ket().p,
        ctx.sampler.samp.max_ket_site(),
        false
    );
}

auto build_density_projected_row(
    qnpeps_ctx& ctx,
    const SamplerConfig& config,
    int row,
    int input_environment_index,
    int output_environment_index,
    std::span<const int> input_bonds,
    std::vector<int>& output_bonds
) -> bool
{
    return apply_row(
        ctx,
        config,
        row,
        input_environment_index,
        input_bonds,
        output_bonds,
        true,
        config.chi_c,
        config.projected_density_cutoff,
        ctx.sampler.samp.env_above()[static_cast<usize>(output_environment_index)].p,
        ctx.sampler.samp.max_env_above_site(),
        true
    );
}
}
