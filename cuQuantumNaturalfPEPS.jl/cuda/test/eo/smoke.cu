#include "dans_qnpeps_eloc.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

static void* dev_cf(int64_t n_complex, float scale)
{
    auto h = std::vector<float>(static_cast<size_t>(2 * n_complex));
    for (auto i = static_cast<size_t>(0); i < h.size(); ++i)
        h[i] = scale * std::sin(0.1f * static_cast<float>(i + 1));
    void* d{};
    cudaMalloc(&d, h.size() * sizeof(float));
    cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}
static int* dev_i32(const std::vector<int>& h)
{
    int* d{};
    cudaMalloc(&d, h.size() * sizeof(int));
    cudaMemcpy(d, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice);
    return d;
}
static int finite_count(void* dev, int64_t n_complex)
{
    auto h = std::vector<float>(static_cast<size_t>(2 * n_complex));
    cudaMemcpy(h.data(), dev, h.size() * sizeof(float), cudaMemcpyDeviceToHost);
    int f{};
    for (float x : h)
        if (std::isfinite(x)) ++f;
    return f;
}

int main()
{
    QnpepsElocConfig cfg{};
    cfg.struct_size = sizeof(QnpepsElocConfig);
    cfg.lx = 4;
    cfg.ly = 4;
    cfg.dim_phys = 2;
    cfg.dim_bond = 2;
    cfg.chi_eo = 8;
    cfg.meo = 8;
    const auto chi = cfg.chi_eo;
    std::printf(
        "[smoke] %s  L=%dx%d d=%d D=%d chi_eo=%d\n",
        qnpeps_eloc_version(),
        cfg.lx,
        cfg.ly,
        cfg.dim_phys,
        cfg.dim_bond,
        chi
    );
    int fails{};

    const auto n_chains = static_cast<int64_t>(32);
    auto ma = dev_cf(n_chains * chi * chi, 0.3f);
    auto mb = dev_cf(n_chains * chi * chi, 0.2f);
    auto vin = dev_cf(n_chains * chi, 0.5f);
    auto vend = dev_cf(n_chains * chi, 0.4f);
    auto out = dev_cf(n_chains, 0.0f);
    auto st = qnpeps_eloc_chains(
        &cfg,
        n_chains,
        static_cast<const qnpeps_eloc_cbuf*>(ma),
        static_cast<const qnpeps_eloc_cbuf*>(mb),
        static_cast<const qnpeps_eloc_cbuf*>(vin),
        static_cast<const qnpeps_eloc_cbuf*>(vend),
        static_cast<qnpeps_eloc_cbuf*>(out),
        nullptr
    );
    std::printf(
        "[smoke] chains: status=%d (%s)  finite=%d/%lld\n",
        st,
        qnpeps_eloc_strerror(st),
        finite_count(out, n_chains),
        2ll * n_chains
    );
    if (st != QNPEPS_ELOC_OK) ++fails;

    const auto n = static_cast<int64_t>(16);
    const auto slice = chi;
    auto env = dev_cf(n * slice * slice, 0.3f);
    auto slin = dev_cf(n * slice, 0.5f);
    auto gsc = dev_cf(n, 0.7f);
    auto oout = dev_cf(n * slice, 0.0f);
    st = qnpeps_eloc_build_o(
        &cfg,
        n,
        slice,
        static_cast<const qnpeps_eloc_cbuf*>(env),
        static_cast<const qnpeps_eloc_cbuf*>(slin),
        static_cast<const qnpeps_eloc_cbuf*>(gsc),
        static_cast<qnpeps_eloc_cbuf*>(oout),
        nullptr
    );
    std::printf(
        "[smoke] build_o: status=%d (%s)  finite=%d/%lld\n",
        st,
        qnpeps_eloc_strerror(st),
        finite_count(oout, n * slice),
        2ll * n * slice
    );
    if (st != QNPEPS_ELOC_OK) ++fails;

    const auto ns = 8;
    const auto n_blocks = 2;
    const auto compact_np = slice * n_blocks;
    auto crows = dev_cf(static_cast<int64_t>(ns) * compact_np, 0.3f);
    auto d_spins = dev_i32(std::vector<int>(static_cast<size_t>(ns) * n_blocks, 0));
    auto d_boff = dev_i32({0, slice});
    auto d_bsl = dev_i32({slice, slice});
    auto gout = dev_cf(static_cast<int64_t>(ns) * ns, 0.0f);
    st = qnpeps_eloc_gram(
        &cfg,
        ns,
        compact_np,
        n_blocks,
        static_cast<const qnpeps_eloc_cbuf*>(crows),
        d_spins,
        d_boff,
        d_bsl,
        static_cast<qnpeps_eloc_cbuf*>(gout),
        nullptr
    );
    std::printf(
        "[smoke] gram: status=%d (%s)  finite=%d/%d\n",
        st,
        qnpeps_eloc_strerror(st),
        finite_count(gout, static_cast<int64_t>(ns) * ns),
        2 * ns * ns
    );
    if (st != QNPEPS_ELOC_OK) ++fails;

    std::printf(fails == 0 ? "[smoke] PASS (ran end-to-end)\n" : "[smoke] FAIL (%d)\n", fails);
    return fails == 0 ? 0 : 1;
}
