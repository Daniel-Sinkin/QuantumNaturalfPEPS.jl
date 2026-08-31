#pragma once

#include "common.cuh"
#include "core/arena_cursor.cuh"
#include "core/complex.cuh"
#include "core/defer.cuh"
#include "core/session.cuh"
#include "density.cuh"
#include "dtensor.cuh"
#include "eloc_kernels.cuh"
#include "env_build.cuh"
#include "../linalg/eo.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <new>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "permutation.hpp"
#include "projection_kernels.hpp"
#include "arena.hpp"
#include "worker.hpp"
#include "environment_rows.hpp"
#include "tensor_network.hpp"
#include "active_lists.hpp"
#include "term_kernels.hpp"
#include "term_evaluation.hpp"
#include "outputs.hpp"

namespace qn_eloc::env
{

#line 3984 "cuda/eo/env_build.cu"

auto classify_route(
    const QnpepsElocConfig& cfg, const QnpepsElocFlipTerm& ft, FlipInst& out, int& pass
) -> bool
{
    const auto lx = int{cfg.lx};
    const auto ly = int{cfg.ly};
    if (ft.n_flips < 1 or ft.n_flips > 4) return false;
    auto minr = int{ft.flip_site[0] / ly};
    auto maxr = int{minr};
    auto minc = int{ft.flip_site[0] % ly};
    auto maxc = int{minc};
    for (auto k = int{0}; k < ft.n_flips; ++k)
    {
        const auto r = int{ft.flip_site[k] / ly};
        const auto c = int{ft.flip_site[k] % ly};
        minr = std::min(minr, r);
        maxr = std::max(maxr, r);
        minc = std::min(minc, c);
        maxc = std::max(maxc, c);
    }
    const auto dx = int{maxr - minr};
    const auto dy = int{maxc - minc};
    out.n_flips = ft.n_flips;
    for (auto k = int{0}; k < ft.n_flips; ++k)
    {
        out.site[static_cast<usize>(k)] = ft.flip_site[k];
        out.value[static_cast<usize>(k)] = ft.flip_value[k];
    }
    out.mask_a = ft.mask_a;
    out.mask_b = ft.mask_b;
    out.coeff_re = ft.coeff_re;
    out.coeff_im = ft.coeff_im;

    if (ft.n_flips == 1)
    {
        pass = 0;
        out.bucket = Bucket::horizontal;
    }
    else if (dx == 0)
    {
        pass = 0;
        out.bucket = dy <= 1 ? Bucket::horizontal : Bucket::longer_horizontal;
    }
    else if (dy == 0)
    {
        pass = 1;
        out.bucket = dx <= 1 ? Bucket::horizontal : Bucket::longer_horizontal;
    }
    else if (dx == 1 and dy == 1 and ft.n_flips == 2)
    {
        pass = 0;
        out.bucket = Bucket::fourbody;
        out.j2_group = minr & 1;
        out.j2_column_group = minc & 1;
    }
    else
    {
        return false;
    }

    if (pass == 1)
    {
        for (auto k = int{0}; k < ft.n_flips; ++k)
        {
            const auto r = int{out.site[static_cast<usize>(k)] / ly};
            const auto c = int{out.site[static_cast<usize>(k)] % ly};
            out.site[static_cast<usize>(k)] = c * lx + r;
        }
        if (out.mask_a >= 0)
        {
            out.mask_a = (out.mask_a % ly) * lx + out.mask_a / ly;
            out.mask_b = (out.mask_b % ly) * lx + out.mask_b / ly;
        }
    }
    return true;
}

}
