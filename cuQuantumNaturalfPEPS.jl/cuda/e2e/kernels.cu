#include "kernels.cuh"

namespace qn_e2e
{

namespace
{

struct ThetaCoordinate
{
    i32 physical{};
    i32 west{};
    i32 north{};
    i32 south{};
    i32 east{};
};

__device__ inline auto take_axis(i64& linear_index, i32 extent) -> i32
{
    const i32 coordinate{static_cast<i32>(linear_index % extent)};
    linear_index /= extent;
    return coordinate;
}

__device__ inline auto decode_theta_coordinate(const UpdateSite& site, i64 theta_local)
    -> ThetaCoordinate
{
    i64 index{theta_local};
    ThetaCoordinate coordinate{};
    coordinate.physical = take_axis(index, site.dp);
    coordinate.west = take_axis(index, site.dw);
    coordinate.north = take_axis(index, site.dn);
    coordinate.south = take_axis(index, site.ds);
    coordinate.east = static_cast<i32>(index);
    return coordinate;
}

__device__ inline auto fixture_local(const UpdateSite& site, const ThetaCoordinate& coordinate)
    -> i64
{
    i64 index{coordinate.physical};
    index = coordinate.north + static_cast<i64>(site.dn) * index;
    index = coordinate.east + static_cast<i64>(site.de) * index;
    index = coordinate.south + static_cast<i64>(site.ds) * index;
    index = coordinate.west + static_cast<i64>(site.dw) * index;
    return index;
}

}

__global__ auto cu_update_f32(UpdateF32Args args) -> void
{
    const UpdateSite site{args.sites[blockIdx.x]};
    for (i64 theta_local{threadIdx.x}; theta_local < site.count; theta_local += blockDim.x)
    {
        const i64 theta_index{site.offset + theta_local};
        const ThetaCoordinate coordinate{decode_theta_coordinate(site, theta_local)};
        const i64 peps_index{site.offset + fixture_local(site, coordinate)};
        const cf direction{args.theta[theta_index]};
        cf value{args.peps[peps_index]};
        value.re = __fadd_rn(value.re, __fmul_rn(args.rate, direction.re));
        value.im = __fadd_rn(value.im, __fmul_rn(args.rate, direction.im));
        args.peps[peps_index] = value;
    }
}

__global__ auto cu_update_f64(UpdateF64Args args) -> void
{
    const UpdateSite site{args.sites[blockIdx.x]};
    for (i64 theta_local{threadIdx.x}; theta_local < site.count; theta_local += blockDim.x)
    {
        const i64 theta_index{site.offset + theta_local};
        const ThetaCoordinate coordinate{decode_theta_coordinate(site, theta_local)};
        const i64 peps_index{site.offset + fixture_local(site, coordinate)};
        const cf direction{args.theta[theta_index]};
        zd value{args.state[peps_index]};
        value.re = __dadd_rn(value.re, __dmul_rn(args.rate, static_cast<f64>(direction.re)));
        value.im = __dadd_rn(value.im, __dmul_rn(args.rate, static_cast<f64>(direction.im)));
        args.state[peps_index] = value;
        args.peps[peps_index] = cf{static_cast<f32>(value.re), static_cast<f32>(value.im)};
    }
}

}
