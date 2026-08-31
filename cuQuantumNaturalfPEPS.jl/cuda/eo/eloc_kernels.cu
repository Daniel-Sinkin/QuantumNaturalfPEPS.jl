#include "eloc_kernels.cuh"

#include <cstdlib>
#include <cstring>

namespace qn_eloc
{
using namespace fx;

__global__ void cu_eloc_chain(
    i64 n_chains,
    int chi,
    const cf* __restrict__ ma,
    const cf* __restrict__ mb,
    const cf* __restrict__ vin,
    const cf* __restrict__ vend,
    cf* __restrict__ out
)
{
    extern __shared__ qnpeps::f32 smem[];
    auto work = reinterpret_cast<cf*>(smem);
    auto partial = work + chi;
    for (auto b = static_cast<i64>(blockIdx.x); b < n_chains; b += gridDim.x)
    {
        auto a = ma + b * chi * chi;
        auto bb = mb + b * chi * chi;
        auto vi = vin + b * chi;
        auto ve = vend + b * chi;
        for (auto row = static_cast<int>(threadIdx.x); row < chi; row += blockDim.x)
        {
            auto acc = cf{};
            for (auto c = 0; c < chi; ++c)
                cf_acc(acc, a[row * chi + c], vi[c]);
            work[row] = acc;
        }
        __syncthreads();
        auto local = cf{};
        for (auto row = static_cast<int>(threadIdx.x); row < chi; row += blockDim.x)
        {
            auto acc = cf{};
            for (auto c = 0; c < chi; ++c)
                cf_acc(acc, bb[row * chi + c], work[c]);
            cf_acc(local, acc, ve[row]);
        }
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

__global__ void cu_build_o(
    i64 n,
    int slice_dim,
    const cf* __restrict__ env,
    const cf* __restrict__ slice_in,
    const cf* __restrict__ gscale,
    cf* __restrict__ out
)
{
    for (auto b = static_cast<i64>(blockIdx.x); b < n; b += gridDim.x)
    {
        auto m = env + b * slice_dim * slice_dim;
        auto v = slice_in + b * slice_dim;
        const auto g = static_cast<cf>(gscale[b]);
        auto o = out + b * slice_dim;
        for (auto row = static_cast<int>(threadIdx.x); row < slice_dim; row += blockDim.x)
        {
            auto acc = cf{};
            for (auto c = 0; c < slice_dim; ++c)
                cf_acc(acc, m[row * slice_dim + c], v[c]);
            o[row] = qnpeps::to_cf(cuCmulf(qnpeps::to_cu(acc), qnpeps::to_cu(g)));
        }
    }
}

__global__ void cu_gram_scalar(
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
    const auto si = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const auto ui = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (si >= s_len or ui >= u_len) return;
    const auto s = s0 + si;
    const auto u = u0 + ui;
    const auto acc = gram_pair_compact(
        rows_a + static_cast<i64>(s) * compact_np,
        rows_b + static_cast<i64>(u) * compact_np,
        spins_a + static_cast<i64>(s) * n_blocks,
        spins_b + static_cast<i64>(u) * n_blocks,
        block_offset,
        block_slice,
        n_blocks
    );
    t[static_cast<i64>(s) * out_ld + u] = acc;
}

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

        const auto matching = static_cast<bool>(active and spin_a[ty] == spin_b[tx]);
        for (auto base = 0; base < site_slice; base += ktile)
        {
            for (auto item = tid; item < tile_y * ktile; item += threads)
            {
                const auto row = item / ktile;
                const auto q = item % ktile;
                const auto sa = static_cast<int>(blockIdx.y) * tile_y + row;
                if (q < site_slice - base and sa < s_len)
                {
                    a[row][q] =
                        rows_a[static_cast<i64>(s0 + sa) * compact_np + site_offset + base + q];
                }
            }
            for (auto item = tid; item < tile_x * ktile; item += threads)
            {
                const auto row = item / ktile;
                const auto q = item % ktile;
                const auto ub = static_cast<int>(blockIdx.x) * tile_x + row;
                if (q < site_slice - base and ub < u_len)
                {
                    b[q][row] =
                        rows_b[static_cast<i64>(u0 + ub) * compact_np + site_offset + base + q];
                }
            }
            __syncthreads();
            if (matching)
            {
                const auto count = static_cast<int>(min(ktile, site_slice - base));
                for (auto q = 0; q < count; ++q)
                    cf_acc_conj(acc, a[ty][q], b[q][tx]);
            }
            __syncthreads();
        }
    }
    if (active) t[static_cast<i64>(s) * out_ld + u] = acc;
}

enum class GramTileMode
{
    tile_16x16,
    tile_8x16,
    tile_16x8,
    tile_8x8
};

static auto gram_tile_mode() -> GramTileMode
{
    static const auto mode = []
    {
        auto value = static_cast<const char*>(std::getenv("QNPEPS_ELOC_GRAM_TILE"));
        if (value == nullptr or std::strcmp(value, "16x16") == 0) return GramTileMode::tile_16x16;
        if (std::strcmp(value, "8x16") == 0) return GramTileMode::tile_8x16;
        if (std::strcmp(value, "16x8") == 0) return GramTileMode::tile_16x8;
        if (std::strcmp(value, "8x8") == 0) return GramTileMode::tile_8x8;
        return GramTileMode::tile_16x16;
    }();
    return mode;
}

static unsigned grid_for(i64 n)
{
    const auto cap = static_cast<i64>(65535);
    const auto g = n < 1 ? 1 : (n < cap ? n : cap);
    return static_cast<unsigned>(g);
}

void launch_eloc_chains(
    i64 n_chains,
    int chi,
    const cf* ma,
    const cf* mb,
    const cf* vin,
    const cf* vend,
    cf* out,
    cudaStream_t stream
)
{
    if (n_chains < 1) return;
    const auto block = 128;
    const auto shmem = (static_cast<size_t>(chi) + block) * sizeof(cf);
    cu_eloc_chain<<<grid_for(n_chains), block, shmem, stream>>>(
        n_chains, chi, ma, mb, vin, vend, out
    );
}

void launch_build_o(
    i64 n,
    int slice_dim,
    const cf* env,
    const cf* slice_in,
    const cf* gscale,
    cf* out,
    cudaStream_t stream
)
{
    if (n < 1) return;
    const auto block = 128;
    cu_build_o<<<grid_for(n), block, 0, stream>>>(n, slice_dim, env, slice_in, gscale, out);
}

void launch_gram(
    int ns,
    int compact_np,
    int n_blocks,
    const cf* compact_rows,
    const int* spins,
    const int* block_offset,
    const int* block_slice,
    cf* out,
    cudaStream_t stream
)
{
    launch_gram_tile(
        out,
        ns,
        compact_np,
        n_blocks,
        compact_rows,
        compact_rows,
        spins,
        spins,
        block_offset,
        block_slice,
        0,
        ns,
        0,
        ns,
        stream
    );
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
    static const auto scalar = []
    {
        auto value = static_cast<const char*>(std::getenv("QNPEPS_ELOC_GRAM_SCALAR"));
        return value != nullptr and *value != '\0' and *value != '0';
    }();
    if (scalar)
    {
        const auto block = dim3(16, 16);
        const auto grid = dim3(
            static_cast<unsigned>((u_len + 15) / 16), static_cast<unsigned>((s_len + 15) / 16)
        );
        cu_gram_scalar<<<grid, block, 0, stream>>>(
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
        return;
    }

    const auto launch_tiled = [&]<int tile_y, int tile_x>()
    {
        const auto block = dim3(tile_x, tile_y);
        const auto grid = dim3(
            static_cast<unsigned>((u_len + tile_x - 1) / tile_x),
            static_cast<unsigned>((s_len + tile_y - 1) / tile_y)
        );
        cu_gram_tiled<tile_y, tile_x><<<grid, block, 0, stream>>>(
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
    };

    switch (gram_tile_mode())
    {
        case GramTileMode::tile_8x16:
            launch_tiled.template operator()<8, 16>();
            break;
        case GramTileMode::tile_16x8:
            launch_tiled.template operator()<16, 8>();
            break;
        case GramTileMode::tile_8x8:
            launch_tiled.template operator()<8, 8>();
            break;
        case GramTileMode::tile_16x16:
            launch_tiled.template operator()<16, 16>();
            break;
    }
}

}
