#include "parts.cuh"

namespace qn_eloc::env
{

__global__ auto cu_project_site_gather(CuProjectGatherArgs args) -> void
{
    const auto slab = static_cast<usize>(args.slab);
    const auto total = slab * static_cast<usize>(args.active);
    const auto grid_stride = static_cast<usize>(gridDim.x) * blockDim.x;
    const auto lane_stride = static_cast<usize>(args.lane_stride);
    const auto out_stride = static_cast<usize>(args.out_stride);
    for (auto tid = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x; tid < total;
         tid += grid_stride)
    {
        const auto compact_lane = tid / slab;
        const auto elem = tid % slab;
        const auto lane = static_cast<usize>(args.active_indices[compact_lane]);
        const auto lane_sample = args.samples + lane * lane_stride;
        auto value = static_cast<usize>(lane_sample[args.site_index]);
        if (args.support_position != k_keep_position)
        {
            const auto state = static_cast<usize>(local_state_index(args.local_state, lane_sample));
            const auto position = static_cast<usize>(args.support_position);
            value =
                args.targets[state * static_cast<usize>(args.local_state.site_count) + position];
        }
        args.out[compact_lane * out_stride + elem] = args.site[value * slab + elem];
    }
}

__global__ auto cu_project_site_weighted(CuProjectWeightedArgs args) -> void
{
    const auto slab = static_cast<usize>(args.slab);
    const auto total = slab * static_cast<usize>(args.active);
    const auto grid_stride = static_cast<usize>(gridDim.x) * blockDim.x;
    const auto lane_stride = static_cast<usize>(args.lane_stride);
    const auto out_stride = static_cast<usize>(args.out_stride);
    const auto dim_phys = static_cast<usize>(args.local_state.dim_phys);
    const auto site_count = static_cast<usize>(args.local_state.site_count);
    const auto position = static_cast<usize>(args.support_position);
    for (auto tid = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x; tid < total;
         tid += grid_stride)
    {
        const auto compact_lane = tid / slab;
        const auto elem = tid % slab;
        const auto lane = static_cast<usize>(args.active_indices[compact_lane]);
        const auto lane_sample = args.samples + lane * lane_stride;
        const auto state = static_cast<usize>(local_state_index(args.local_state, lane_sample));
        const auto base = (state * site_count + position) * dim_phys;
        auto weighted = cf{};
        for (auto value = 0_uz; value < dim_phys; ++value)
            fx::cf_acc(weighted, args.weights[base + value], args.site[value * slab + elem]);
        args.out[compact_lane * out_stride + elem] = weighted;
    }
}

__global__ auto cu_compact_lanes(
    cf* out,
    i64 out_stride,
    const cf* in,
    i64 in_stride,
    const int* active_indices,
    i64 elems,
    int active
) -> void
{
    const auto total = i64{elems * active};
    const auto grid_stride = i64{static_cast<i64>(gridDim.x) * blockDim.x};
    for (auto tid = static_cast<i64>(blockIdx.x * blockDim.x + threadIdx.x); tid < total;
         tid += grid_stride)
    {
        const auto compact_lane = int{static_cast<int>(tid / elems)};
        const auto elem = i64{tid % elems};
        const auto lane = int{active_indices[compact_lane]};
        out[static_cast<i64>(compact_lane) * out_stride + elem] =
            in[static_cast<i64>(lane) * in_stride + elem];
    }
}

__global__ auto cu_diagonal(CuDiagonalArgs args) -> void
{
    const auto lane = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (lane >= static_cast<usize>(args.lanes)) return;
    const auto lane_sample = args.samples + lane * static_cast<usize>(args.lane_stride);
    const auto term_count = static_cast<usize>(args.term_count);
    auto energy_real = 0.0;
    auto energy_imag = 0.0;
    for (auto t = 0_uz; t < term_count; ++t)
    {
        const auto& term = args.terms[t];
        const auto state = local_state_index(term.local_state, lane_sample);
        const auto value = args.values[term.value_offset + state];
        energy_real += value.x;
        energy_imag += value.y;
    }
    args.e_loc[2 * lane] += energy_real;
    args.e_loc[2 * lane + 1] += energy_imag;
}

__global__ auto cu_reduce_term_indexed(CuReduceTermArgs args) -> void
{
    const auto compact_lane = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (compact_lane >= static_cast<usize>(args.active)) return;
    const auto v = args.value[compact_lane];
    if (v.re == 0.0f and v.im == 0.0f) return;
    const auto lane = static_cast<usize>(args.active_indices[compact_lane]);
    const auto lane_sample = args.samples + lane * static_cast<usize>(args.lane_stride);
    const auto coefficient = args.coefficients[local_state_index(args.local_state, lane_sample)];
    const auto coeff_re = coefficient.x;
    const auto coeff_im = coefficient.y;
    const auto re = static_cast<f64>(v.re);
    const auto im = static_cast<f64>(v.im);
    const f64 logmag{0.5 * log(norm2(re, im))};
    const f64 phase{atan2(im, re)};
    const f64 fsum{
        (args.f_top ? args.f_top[lane] : 0.0) + (args.f_down ? args.f_down[lane] : 0.0)
        + (args.f_rail_l ? args.f_rail_l[lane] : 0.0) + (args.f_rail_r ? args.f_rail_r[lane] : 0.0)
        + args.f_chain[compact_lane]
    };
    const f64 dre{logmag + fsum - args.logpsi[2 * lane]};
    const f64 dim{phase - args.logpsi[2 * lane + 1]};
    const f64 mag{exp(dre)};
    const f64 rr{mag * cos(dim)};
    const f64 ri{mag * sin(dim)};
    args.e_loc[2 * lane] += coeff_re * rr - coeff_im * ri;
    args.e_loc[2 * lane + 1] += coeff_re * ri + coeff_im * rr;
}

auto reduce_term_indexed(
    Worker& w,
    const TermPiece& piece,
    const cf* value,
    const f64* f_chain,
    const int* active_indices,
    int active,
    f64* f_top,
    f64* f_down,
    f64* f_rail_l,
    f64* f_rail_r,
    f64* e_loc,
    const f64* logpsi
) -> void
{
    const auto blocks = (active + 255) / 256;
    const auto args = CuReduceTermArgs{
        .e_loc = e_loc,
        .value = value,
        .f_top = f_top,
        .f_down = f_down,
        .f_rail_l = f_rail_l,
        .f_rail_r = f_rail_r,
        .f_chain = f_chain,
        .logpsi = logpsi,
        .active_indices = active_indices,
        .samples = w.samples,
        .lane_stride = static_cast<i64>(w.sh.lx) * w.sh.ly,
        .local_state = piece.local_state,
        .coefficients = piece.device_coefficients,
        .active = active,
    };
    cu_reduce_term_indexed<<<blocks, 256, 0, w.la.stream()>>>(args);
    check(cudaGetLastError());
}

auto launch_diagonal(Worker& w, f64* e_loc, const DeviceTermPools& pools, int term_count) -> void
{
    if (term_count < 1) return;
    const auto lanes = w.sh.lanes;
    const auto blocks = (lanes + 255) / 256;
    const auto args = CuDiagonalArgs{
        .e_loc = e_loc,
        .samples = w.samples,
        .lane_stride = static_cast<i64>(w.sh.lx) * w.sh.ly,
        .terms = pools.diagonal,
        .values = pools.diagonal_values,
        .term_count = term_count,
        .lanes = lanes,
    };
    cu_diagonal<<<blocks, 256, 0, w.la.stream()>>>(args);
    check(cudaGetLastError());
}

__global__ auto cu_fourbody(
    int lanes, int mid, const cf* cl, i64 stride_l, const cf* cr, i64 stride_r, cf* out
) -> void
{
    extern __shared__ f32 smem[];
    auto partial{reinterpret_cast<cf*>(smem)};
    for (auto b = static_cast<int>(blockIdx.x); b < lanes; b += gridDim.x)
    {
        auto a{cl + static_cast<i64>(b) * stride_l};
        auto bb{cr + static_cast<i64>(b) * stride_r};
        auto local = cf{};
        for (auto k = static_cast<int>(threadIdx.x); k < mid; k += blockDim.x)
            fx::cf_acc(local, a[k], bb[k]);
        partial[threadIdx.x] = local;
        __syncthreads();
        for (auto h = static_cast<int>(blockDim.x / 2); h > 0; h >>= 1)
        {
            if (static_cast<int>(threadIdx.x) < h)
                partial[threadIdx.x] = qnpeps::to_cf(cuCaddf(
                    qnpeps::to_cu(partial[threadIdx.x]), qnpeps::to_cu(partial[threadIdx.x + h])
                ));
            __syncthreads();
        }
        if (threadIdx.x == 0) out[b] = partial[0];
    }
}

auto launch_dot(
    Worker& w,
    int mid,
    const cf* cl,
    i64 stride_l,
    const cf* cr,
    i64 stride_r,
    cf* out,
    int dim_batch
) -> void
{
    const auto lanes = int{dim_batch > 0 ? dim_batch : w.sh.lanes};
    const auto block = int{128};
    const auto shmem = usize{static_cast<usize>(block) * sizeof(cf)};
    const auto grid = int{std::min(lanes, 65535)};
    cu_fourbody<<<grid, block, shmem, w.la.stream()>>>(lanes, mid, cl, stride_l, cr, stride_r, out);
    check(cudaGetLastError());
}

}
