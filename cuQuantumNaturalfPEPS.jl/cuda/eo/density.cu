#include "core/arena_cursor.cuh"
#include "core/complex.cuh"
#include "core/cuda_utils.cuh"
#include "densitymatrix/backend.cuh"
#include "densitymatrix/types.cuh"
#include "eo/density.cuh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <new>
#include <span>
#include <vector>

namespace qn_eloc::density
{
namespace
{
using qnpeps::f64;
using qnpeps::i32;
using qnpeps::i64;
using qnpeps::usize;
using namespace qnpeps;

struct ProductContext
{
    const RowArgs* args{};
};

struct ProductArgs
{
    const cuDoubleComplex* state{};
    const qn_eloc::fx::cf* projected{};
    i64 state_left{};
    i64 state_right{};
    i64 west{};
    i64 south{};
    i64 east{};
    i64 north{};
    i64 output_state{};
    bool contract_up{};
    cuDoubleComplex* product{};
};

struct PromoteArgs
{
    const qn_eloc::fx::cf* input{};
    usize count{};
    cuDoubleComplex* output{};
};

struct PackArgs
{
    const cuDoubleComplex* input{};
    i64 physical{};
    i64 right{};
    i64 left{};
    qn_eloc::fx::cf* output{};
};

struct BoundaryArgs
{
    const qn_eloc::fx::cf* input{};
    qn_eloc::fx::cf* output{};
    i64 west{};
    i64 south{};
    i64 east{};
    i64 north{};
    bool contract_up{};
    f64* gauge{};
};

__global__ auto cu_product(ProductArgs args) -> void
{
    const auto operator_left = args.west;
    const auto operator_right = args.east;
    const auto combined_left = args.state_left * operator_left;
    const auto combined_right = args.state_right * operator_right;
    const auto count = combined_left * combined_right;
    for (auto lane = qnpeps::global_lane(); lane < count; lane += qnpeps::grid_stride())
    {
        const auto combined_left_index = lane % combined_left;
        const auto combined_right_index = lane / combined_left;
        const auto state_left = combined_left_index % args.state_left;
        const auto west = combined_left_index / args.state_left;
        const auto state_right = combined_right_index % args.state_right;
        const auto east = combined_right_index / args.state_right;
        const auto physical_input = args.contract_up ? args.north : args.south;
        auto value = make_cuDoubleComplex(0.0, 0.0);
        for (auto physical = 0_i64; physical < physical_input; ++physical)
        {
            const auto state_index =
                state_left + args.state_left * (physical + physical_input * state_right);
            const auto south = args.contract_up ? args.output_state : physical;
            const auto north = args.contract_up ? physical : args.output_state;
            const auto projected_index =
                west + args.west * (south + args.south * (east + args.east * north));
            const auto input = args.projected[projected_index];
            value = cuCadd(
                value,
                cuCmul(
                    args.state[state_index],
                    make_cuDoubleComplex(static_cast<f64>(input.re), static_cast<f64>(input.im))
                )
            );
        }
        args.product[lane] = value;
    }
}

__global__ auto cu_promote(PromoteArgs args) -> void
{
    for (auto lane = qnpeps::global_lane(); lane < static_cast<i64>(args.count);
         lane += qnpeps::grid_stride())
    {
        const auto input = args.input[lane];
        args.output[lane] =
            make_cuDoubleComplex(static_cast<f64>(input.re), static_cast<f64>(input.im));
    }
}

__global__ auto cu_pack(PackArgs args) -> void
{
    const auto count = args.left * args.physical * args.right;
    for (auto lane = qnpeps::global_lane(); lane < count; lane += qnpeps::grid_stride())
    {
        const auto left = lane % args.left;
        const auto tail = lane / args.left;
        const auto physical = tail % args.physical;
        const auto right = tail / args.physical;
        const auto value = args.input[physical + args.physical * (right + args.right * left)];
        args.output[lane] = to_cf(value);
    }
}

__global__ auto cu_boundary(BoundaryArgs args) -> void
{
    __shared__ qnpeps::CuArray<f64, 256> reduction;
    const auto vertical = args.contract_up ? args.south : args.north;
    const auto count = args.west * vertical * args.east;
    auto sum = 0.0;
    for (auto index = static_cast<i64>(threadIdx.x); index < count; index += blockDim.x)
    {
        const auto west = index % args.west;
        const auto tail = index / args.west;
        const auto output_state = tail % vertical;
        const auto east = tail / vertical;
        const auto south = args.contract_up ? output_state : 0_i64;
        const auto north = args.contract_up ? 0_i64 : output_state;
        const auto source = west + args.west * (south + args.south * (east + args.east * north));
        const auto value = args.input[source];
        sum += static_cast<f64>(value.re) * value.re + static_cast<f64>(value.im) * value.im;
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (auto stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const auto norm = sqrt(reduction[0]);
    const auto scale = norm > 0.0 ? 1.0 / norm : 1.0;
    for (auto index = static_cast<i64>(threadIdx.x); index < count; index += blockDim.x)
    {
        const auto west = index % args.west;
        const auto tail = index / args.west;
        const auto output_state = tail % vertical;
        const auto east = tail / vertical;
        const auto south = args.contract_up ? output_state : 0_i64;
        const auto north = args.contract_up ? 0_i64 : output_state;
        const auto source = west + args.west * (south + args.south * (east + args.east * north));
        const auto value = args.input[source];
        args.output[index] =
            qn_eloc::fx::cf{static_cast<f32>(value.re * scale), static_cast<f32>(value.im * scale)};
    }
    if (threadIdx.x == 0 and norm > 0.0) *args.gauge += log(norm);
}

auto product(
    qnpeps::Linalg& linalg,
    const qnpeps::densitymatrix::Site& site,
    i64 output_state,
    const cuDoubleComplex* state,
    const cuDoubleComplex*,
    cuDoubleComplex* output,
    const void* context_value
) -> void
{
    const auto& context = *static_cast<const ProductContext*>(context_value);
    const auto& args = *context.args;
    const auto index = static_cast<usize>(site.site);
    if (index >= static_cast<usize>(args.sites))
        return qnpeps::set_err(QNPEPS_ERR_INTERNAL), void();
    const auto& shape = args.shapes[index];
    const auto combined_left = site.state_left * site.operator_left;
    const auto combined_right = site.state_right * site.operator_right;
    cu_product<<<
        qnpeps::grid_blocks_capped(combined_left * combined_right),
        qnpeps::k_threads_per_block,
        0,
        linalg.stream()>>>(
        {.state = state + site.state_offset,
         .projected = args.projected[index],
         .state_left = site.state_left,
         .state_right = site.state_right,
         .west = shape.west,
         .south = shape.south,
         .east = shape.east,
         .north = shape.north,
         .output_state = output_state,
         .contract_up = args.contract_up,
         .product = output}
    );
    CUDA_CHECK(cudaGetLastError());
}

auto status_value() -> int
{
    return static_cast<int>(qnpeps::err_state());
}
}

struct Context
{
    qnpeps::Linalg* linalg{};
    qnpeps::densitymatrix::Workspace workspace{};
    cuDoubleComplex* state_values{};
    usize state_count{};
    cuDoubleComplex* result_values{};
    usize result_count{};
    i32* result_dimensions{};
    f64* normalization_log{};
    f64* output_gauge{};
    usize result_slot{};
    int sites{};
    int dim_bond{};
    int max_bond{};
};

auto create(
    qnpeps::Linalg& linalg,
    qnpeps::ArenaCursor& arena,
    int sites,
    int dim_bond,
    int max_bond,
    Context** output
) -> int
{
    qnpeps::reset_err();
    if (not output) return static_cast<int>(QNPEPS_ERR_NULL_ARG);
    auto context = std::unique_ptr<Context>{new (std::nothrow) Context{}};
    if (not context) return static_cast<int>(QNPEPS_ERR_OOM);
    context->linalg = &linalg;
    if (sites < 2 or dim_bond < 1 or max_bond < 1) return static_cast<int>(QNPEPS_ERR_BAD_CONFIG);
    const auto state_slot =
        static_cast<usize>(max_bond) * static_cast<usize>(dim_bond) * static_cast<usize>(max_bond);
    context->state_count = static_cast<usize>(sites) * state_slot;
    context->result_count = context->state_count;
    context->workspace = qnpeps::densitymatrix::take_workspace(
        *context->linalg,
        arena,
        {.num_sites = sites,
         .input_bond = max_bond,
         .operator_bond = dim_bond,
         .output_dimension = dim_bond,
         .upper_bond = max_bond,
         .lanes = 1}
    );
    context->state_values = arena.take<cuDoubleComplex>(context->state_count);
    context->result_values = arena.take<cuDoubleComplex>(context->result_count);
    context->result_dimensions = arena.take<i32>(3_uz * static_cast<usize>(sites));
    context->normalization_log = arena.take<f64>(1);
    context->output_gauge = arena.take<f64>(1);
    context->result_slot = state_slot;
    context->sites = sites;
    context->dim_bond = dim_bond;
    context->max_bond = max_bond;
    if (qnpeps::err_state() != QNPEPS_OK) return status_value();
    *output = context.release();
    return static_cast<int>(QNPEPS_OK);
}

auto destroy(Context* context) -> void
{
    delete context;
}

auto boundary(Context& context, const RowArgs& args) -> int
{
    qnpeps::reset_err();
    if (args.sites != context.sites or not args.shapes or not args.projected or not args.output
        or not args.output_ranks or not args.device_gauge)
        return static_cast<int>(QNPEPS_ERR_BAD_CONFIG);
    args.output_ranks[0] = 1;
    for (auto site = 0; site < args.sites; ++site)
    {
        const auto& shape = args.shapes[site];
        const auto vertical = args.contract_up ? shape.south : shape.north;
        if ((args.contract_up and shape.north != 1) or (not args.contract_up and shape.south != 1)
            or shape.west < 1 or vertical < 1 or shape.east < 1)
            return static_cast<int>(QNPEPS_ERR_BAD_CONFIG);
        if (site > 0 and args.output_ranks[site] != shape.west)
            return static_cast<int>(QNPEPS_ERR_INTERNAL);
        args.output_ranks[site] = static_cast<int>(shape.west);
        args.output_ranks[site + 1] = static_cast<int>(shape.east);
        cu_boundary<<<1, qnpeps::k_threads_per_block, 0, context.linalg->stream()>>>(
            {.input = args.projected[site],
             .output = args.output[site],
             .west = shape.west,
             .south = shape.south,
             .east = shape.east,
             .north = shape.north,
             .contract_up = args.contract_up,
             .gauge = args.device_gauge}
        );
        CUDA_CHECK(cudaGetLastError());
    }
    if (args.output_ranks[0] != 1 or args.output_ranks[args.sites] != 1)
        return static_cast<int>(QNPEPS_ERR_INTERNAL);
    return status_value();
}

auto contract(Context& context, const RowArgs& args) -> int
{
    qnpeps::reset_err();
    if (args.sites != context.sites or args.max_bond != context.max_bond
        or not std::isfinite(args.cutoff) or args.cutoff < 0.0 or not args.shapes
        or not args.projected or not args.adjacent or not args.output or not args.adjacent_ranks
        or not args.output_ranks or not args.device_gauge)
        return static_cast<int>(QNPEPS_ERR_BAD_CONFIG);
    auto input_gauge = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(
        &input_gauge,
        args.device_gauge,
        sizeof(input_gauge),
        cudaMemcpyDeviceToHost,
        context.linalg->stream()
    ));
    CUDA_CHECK(cudaStreamSynchronize(context.linalg->stream()));
    if (qnpeps::err_state() != QNPEPS_OK) return status_value();
    auto sites = std::vector<qnpeps::densitymatrix::Site>{};
    sites.reserve(static_cast<usize>(args.sites));
    auto state_offset = 0_uz;
    for (auto site = 0; site < args.sites; ++site)
    {
        const auto& shape = args.shapes[site];
        const auto left = static_cast<i64>(args.adjacent_ranks[site]);
        const auto right = static_cast<i64>(args.adjacent_ranks[site + 1]);
        const auto contracted = args.contract_up ? shape.north : shape.south;
        const auto vertical = args.contract_up ? shape.south : shape.north;
        const auto state_size = static_cast<usize>(left * contracted * right);
        cu_promote<<<
            qnpeps::grid_blocks_capped(static_cast<i64>(state_size)),
            qnpeps::k_threads_per_block,
            0,
            context.linalg->stream()>>>(
            {.input = args.adjacent[site],
             .count = state_size,
             .output = context.state_values + state_offset}
        );
        sites.push_back(
            {.site = static_cast<usize>(site),
             .num_sites = static_cast<usize>(args.sites),
             .state_left = left,
             .physical_input = contracted,
             .state_right = right,
             .operator_left = shape.west,
             .physical_output = vertical,
             .operator_right = shape.east,
             .state_offset = state_offset,
             .operator_offset = 0,
             .output_offset = static_cast<usize>(site) * context.result_slot}
        );
        state_offset += state_size;
    }
    CUDA_CHECK(cudaGetLastError());
    if (state_offset > context.state_count or qnpeps::err_state() != QNPEPS_OK)
        return static_cast<int>(QNPEPS_ERR_BAD_CONFIG);
    const auto product_context = ProductContext{&args};
    qnpeps::densitymatrix::apply(
        *context.linalg,
        {.settings =
             {.struct_size = static_cast<std::uint32_t>(sizeof(QnpepsDensitySettings)),
              .precision = 1,
              .mindim = 1,
              .reserved = 0,
              .relative_cutoff = args.cutoff},
         .upper_bond = args.max_bond,
         .normalize = true,
         .input_gauge = input_gauge,
         .sites = std::span<const qnpeps::densitymatrix::Site>{sites},
         .workspace = context.workspace,
         .state_values = context.state_values,
         .state_value_count = state_offset,
         .operator_values = nullptr,
         .operator_value_count = 0,
         .result_dimensions = context.result_dimensions,
         .result_dimension_count = 3_uz * static_cast<usize>(args.sites),
         .result_values = context.result_values,
         .result_value_count = context.result_count,
         .normalization_log = context.normalization_log,
         .output_gauge = context.output_gauge,
         .product = product,
         .product_context = &product_context}
    );
    if (qnpeps::err_state() != QNPEPS_OK) return status_value();
    auto dimensions = std::vector<i32>(3_uz * static_cast<usize>(args.sites));
    CUDA_CHECK(cudaMemcpyAsync(
        dimensions.data(),
        context.result_dimensions,
        dimensions.size() * sizeof(i32),
        cudaMemcpyDeviceToHost,
        context.linalg->stream()
    ));
    CUDA_CHECK(cudaMemcpyAsync(
        args.device_gauge,
        context.output_gauge,
        sizeof(f64),
        cudaMemcpyDeviceToDevice,
        context.linalg->stream()
    ));
    CUDA_CHECK(cudaStreamSynchronize(context.linalg->stream()));
    if (qnpeps::err_state() != QNPEPS_OK) return status_value();
    for (auto site = 0; site < args.sites; ++site)
    {
        const auto offset = 3 * site;
        const auto physical = static_cast<i64>(dimensions[static_cast<usize>(offset)]);
        const auto right = static_cast<i64>(dimensions[static_cast<usize>(offset + 1)]);
        const auto left = static_cast<i64>(dimensions[static_cast<usize>(offset + 2)]);
        const auto expected = args.contract_up ? args.shapes[site].south : args.shapes[site].north;
        if (physical != expected or left < 1 or right < 1
            or (site > 0 and args.output_ranks[site] != left))
            return static_cast<int>(QNPEPS_ERR_INTERNAL);
        args.output_ranks[site] = static_cast<int>(left);
        args.output_ranks[site + 1] = static_cast<int>(right);
        cu_pack<<<
            qnpeps::grid_blocks_capped(left * physical * right),
            qnpeps::k_threads_per_block,
            0,
            context.linalg->stream()>>>(
            {.input = context.result_values + static_cast<usize>(site) * context.result_slot,
             .physical = physical,
             .right = right,
             .left = left,
             .output = args.output[site]}
        );
    }
    CUDA_CHECK(cudaGetLastError());
    if (args.output_ranks[0] != 1 or args.output_ranks[args.sites] != 1)
        return static_cast<int>(QNPEPS_ERR_INTERNAL);
    return status_value();
}
}
