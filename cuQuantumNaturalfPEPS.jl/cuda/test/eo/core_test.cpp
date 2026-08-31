#include "eloc_fixed.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace qn_eloc::fx;

static int g_fail = 0;
static void check(bool ok, const char* what)
{
    if (!ok)
    {
        std::printf("  FAIL: %s\n", what);
        ++g_fail;
    }
}
static bool close(cf a, cf b)
{
    return std::fabs(a.re - b.re) < 1e-5f && std::fabs(a.im - b.im) < 1e-5f;
}

int main()
{
    {
        const cf I[4] = {cf{1, 0}, cf{0, 0}, cf{0, 0}, cf{1, 0}};
        const cf x[2] = {cf{1, 0}, cf{2, 0}};
        cf y[2];
        matvec_rm(y, I, x, 2, 2);
        check(close(y[0], cf{1, 0}) && close(y[1], cf{2, 0}), "matvec_rm identity");
    }

    {
        const cf I[4] = {cf{1, 0}, cf{0, 0}, cf{0, 0}, cf{1, 0}};
        const cf vin[2] = {cf{1, 0}, cf{2, 0}};
        const cf vend[2] = {cf{3, 0}, cf{4, 0}};
        cf work[2];
        const cf out = eloc_chain_value(I, I, vin, vend, work, 2);
        check(close(out, cf{11, 0}), "eloc_chain_value (I,I)");
    }

    {
        const cf I[4] = {cf{1, 0}, cf{0, 0}, cf{0, 0}, cf{1, 0}};
        const cf slice_in[2] = {cf{1, 0}, cf{2, 0}};
        cf o[2];
        ok_site_slice(o, I, slice_in, cf{0, 1}, 2);
        check(close(o[0], cf{0, 1}) && close(o[1], cf{0, 2}), "ok_site_slice (scale by i)");
    }

    {
        const cf row_s[1] = {cf{1, 1}};
        const cf row_t[1] = {cf{2, -1}};
        const int spin_s[1] = {0};
        const int spin_t[1] = {0};
        const int block_offset[1] = {0};
        const int block_slice[1] = {1};
        const cf g = gram_pair_compact(row_s, row_t, spin_s, spin_t, block_offset, block_slice, 1);
        check(close(g, cf{1, -3}), "gram_pair_compact (matching spins)");
    }
    {
        const cf row_s[1] = {cf{1, 1}};
        const cf row_t[1] = {cf{2, -1}};
        const int spin_s[1] = {0};
        const int spin_t[1] = {1};
        const int block_offset[1] = {0};
        const int block_slice[1] = {1};
        const cf g = gram_pair_compact(row_s, row_t, spin_s, spin_t, block_offset, block_slice, 1);
        check(close(g, cf{0, 0}), "gram_pair_compact (differing spins -> 0)");
    }

    std::printf(g_fail == 0 ? "[core_test] PASS\n" : "[core_test] FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
