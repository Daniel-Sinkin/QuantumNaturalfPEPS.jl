#pragma once

#include "core/complex.cuh"

#include <cmath>

#if defined(__CUDACC__)
#    define ELOC_HD __host__ __device__
#    define ELOC_FI __forceinline__
#else
#    define ELOC_HD
#    define ELOC_FI inline
#endif

namespace qn_eloc::fx
{

using cf = qnpeps::ComplexF32;

ELOC_HD ELOC_FI qnpeps::f32 cf_abs(cf a)
{
    return std::sqrt(qnpeps::norm2(a));
}
ELOC_HD ELOC_FI void cf_acc(cf& acc, cf a, cf b)
{
    acc.re += a.re * b.re - a.im * b.im;
    acc.im += a.re * b.im + a.im * b.re;
}
ELOC_HD ELOC_FI void cf_acc_conj(cf& acc, cf a, cf b)
{
    acc.re += a.re * b.re + a.im * b.im;
    acc.im += a.re * b.im - a.im * b.re;
}

ELOC_HD ELOC_FI void matvec_rm(cf* y, const cf* a, const cf* x, int M, int K)
{
    for (auto i = 0; i < M; ++i)
    {
        auto acc = cf{};
        for (auto k = 0; k < K; ++k)
            cf_acc(acc, a[i * K + k], x[k]);
        y[i] = acc;
    }
}

ELOC_HD ELOC_FI cf
eloc_chain_value(const cf* ma, const cf* mb, const cf* vin, const cf* vend, cf* work, int chi)
{
    matvec_rm(work, ma, vin, chi, chi);
    auto out = cf{};
    for (auto r = 0; r < chi; ++r)
    {
        auto row = cf{};
        for (auto c = 0; c < chi; ++c)
            cf_acc(row, mb[r * chi + c], work[c]);
        cf_acc(out, row, vend[r]);
    }
    return out;
}

ELOC_HD ELOC_FI void ok_site_slice(
    cf* o_out, const cf* env, const cf* slice_in, cf g, int slice_dim
)
{
    for (auto r = 0; r < slice_dim; ++r)
    {
        auto acc = cf{};
        for (auto c = 0; c < slice_dim; ++c)
            cf_acc(acc, env[r * slice_dim + c], slice_in[c]);
        o_out[r] = qnpeps::to_cf(cuCmulf(qnpeps::to_cu(acc), qnpeps::to_cu(g)));
    }
}

ELOC_HD ELOC_FI cf gram_pair_compact(
    const cf* row_s,
    const cf* row_t,
    const int* spin_s,
    const int* spin_t,
    const int* block_offset,
    const int* block_slice,
    int n_blocks
)
{
    auto acc = cf{};
    for (auto b = 0; b < n_blocks; ++b)
    {
        if (spin_s[b] != spin_t[b]) continue;
        const auto off = block_offset[b];
        const auto len = block_slice[b];
        for (auto k = 0; k < len; ++k)
            cf_acc_conj(acc, row_s[off + k], row_t[off + k]);
    }
    return acc;
}

}
