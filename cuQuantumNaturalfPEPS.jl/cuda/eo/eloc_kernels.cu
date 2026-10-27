#include "eloc_kernels.cuh"

#include <cstdlib>
#include <cstring>

namespace qn_eloc
{
using namespace fx;

template <int tile_y, int tile_x>
__global__ void cu_gram_tiled(
    cf* t,
    int out_ld,
    int compact_np,
    int n_blocks,
    const cf* __restrict__ rows_a,
    const cf* __restrict__ rows_b,
    const int* __restrict__ spins_a,
    const int* __restrict__ spins_b,
    const int* __restrict__ block_offset,
    const int* __restrict__ block_slice,
    int s0,
    int s_len,
    int u0,
    int u_len
)
{
    constexpr auto ktile = 128;
    constexpr auto threads = tile_y * tile_x;
    static_assert(threads <= 1024);
    __shared__ qnpeps::CuArray<qnpeps::CuArray<cf, ktile>, tile_y> a;
    __shared__ qnpeps::CuArray<qnpeps::CuArray<cf, tile_x + 1>, ktile> b;
    __shared__ qnpeps::CuArray<int, tile_y> spin_a;
    __shared__ qnpeps::CuArray<int, tile_x> spin_b;
    __shared__ int site_offset;
    __shared__ int site_slice;

    const auto tx = static_cast<int>(threadIdx.x);
    const auto ty = static_cast<int>(threadIdx.y);
    const auto tid = ty * tile_x + tx;
    const auto si = static_cast<int>(blockIdx.y) * tile_y + ty;
    const auto ui = static_cast<int>(blockIdx.x) * tile_x + tx;
    const auto active = si < s_len and ui < u_len;
    const auto s = s0 + si;
    const auto u = u0 + ui;
    auto acc = cf{};

    for (auto site = 0; site < n_blocks; ++site)
    {
        if (tid == 0)
        {
            site_offset = block_offset[site];
            site_slice = block_slice[site];
        }
        if (tid < tile_y)
        {
            const auto sample = static_cast<int>(blockIdx.y) * tile_y + tid;
            spin_a[tid] =
                sample < s_len ? spins_a[static_cast<i64>(s0 + sample) * n_blocks + site] : 0;
        }
        if (tid >= tile_y and tid < tile_y + tile_x)
        {
            const auto row = tid - tile_y;
            const auto sample = static_cast<int>(blockIdx.x) * tile_x + row;
            spin_b[row] =
                sample < u_len ? spins_b[static_cast<i64>(u0 + sample) * n_blocks + site] : 0;
        }
        __syncthreads();

        const auto offset = site_offset;
        const auto slice = site_slice;
        const auto matching = static_cast<bool>(active and spin_a[ty] == spin_b[tx]);
        for (auto base = 0; base < slice; base += ktile)
        {
            for (auto item = tid; item < tile_y * ktile; item += threads)
            {
                const auto row = item / ktile;
                const auto q = item % ktile;
                const auto sa = static_cast<int>(blockIdx.y) * tile_y + row;
                if (q < slice - base and sa < s_len)
                {
                    a[row][q] = rows_a[static_cast<i64>(s0 + sa) * compact_np + offset + base + q];
                }
            }
            for (auto item = tid; item < tile_x * ktile; item += threads)
            {
                const auto row = item / ktile;
                const auto q = item % ktile;
                const auto ub = static_cast<int>(blockIdx.x) * tile_x + row;
                if (q < slice - base and ub < u_len)
                {
                    b[q][row] = rows_b[static_cast<i64>(u0 + ub) * compact_np + offset + base + q];
                }
            }
            __syncthreads();
            if (matching)
            {
                const auto count = static_cast<int>(min(ktile, slice - base));
                for (auto q = 0; q < count; ++q)
                    cf_acc_conj(acc, a[ty][q], b[q][tx]);
            }
            __syncthreads();
        }
        __syncthreads();
    }
    if (active) t[static_cast<i64>(s) * out_ld + u] = acc;
}

void launch_gram_tile(
    cf* out,
    int out_ld,
    int compact_np,
    int n_blocks,
    const cf* rows_a,
    const cf* rows_b,
    const int* spins_a,
    const int* spins_b,
    const int* block_offset,
    const int* block_slice,
    int s0,
    int s_len,
    int u0,
    int u_len,
    cudaStream_t stream
)
{
    if (s_len < 1 or u_len < 1) return;
    constexpr auto tile = 16;
    const auto block = dim3(tile, tile);
    const auto grid = dim3(
        static_cast<unsigned>((u_len + tile - 1) / tile),
        static_cast<unsigned>((s_len + tile - 1) / tile)
    );
    cu_gram_tiled<tile, tile><<<grid, block, 0, stream>>>(
        out,
        out_ld,
        compact_np,
        n_blocks,
        rows_a,
        rows_b,
        spins_a,
        spins_b,
        block_offset,
        block_slice,
        s0,
        s_len,
        u0,
        u_len
    );
}

}
