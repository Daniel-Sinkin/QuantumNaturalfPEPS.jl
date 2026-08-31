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

namespace qn_eloc::env
{

#line 281 "cuda/eo/env_build.cu"

struct Shape
{
    int lx{};
    int ly{};
    int dim_phys{};
    int dim_bond{};
    int chi{};
    int lanes{};
    bool density{};
    f64 density_cutoff{};
};

auto arena_product(std::initializer_list<std::uint64_t> factors, usize& result) -> bool
{
    auto product = std::uint64_t{1};
    for (const std::uint64_t factor : factors)
    {
        if (factor != 0 and product > std::numeric_limits<std::uint64_t>::max() / factor)
        {
            qn::set_err(QNPEPS_ELOC_ERR_OOM);
            result = 0;
            return false;
        }
        product *= factor;
    }
    if (product > std::numeric_limits<usize>::max())
    {
        qn::set_err(QNPEPS_ELOC_ERR_OOM);
        result = 0;
        return false;
    }
    result = static_cast<usize>(product);
    return true;
}

auto arena_sum(usize left, usize right, usize& result) -> bool
{
    if (right > std::numeric_limits<usize>::max() - left)
    {
        qn::set_err(QNPEPS_ELOC_ERR_OOM);
        result = 0;
        return false;
    }
    result = left + right;
    return true;
}

auto arena_accumulate(usize& total, usize term) -> bool
{
    usize result{};
    if (not arena_sum(total, term, result)) return false;
    total = result;
    return true;
}

auto arena_align(usize value, usize& result) -> bool
{
    usize padded{};
    if (not arena_sum(value, static_cast<usize>(CUDA_MALLOC_ALIGN - 1), padded)) return false;
    result = padded & ~static_cast<usize>(CUDA_MALLOC_ALIGN - 1);
    return true;
}

auto arena_slot(std::initializer_list<std::uint64_t> factors, i64& result) -> bool
{
    usize product{};
    if (not arena_product(factors, product)) return false;
    if (product > static_cast<usize>(std::numeric_limits<i64>::max()))
    {
        qn::set_err(QNPEPS_ELOC_ERR_OOM);
        result = 0;
        return false;
    }
    result = static_cast<i64>(product);
    return true;
}

auto arena_int(std::initializer_list<std::uint64_t> factors, int& result) -> bool
{
    usize product{};
    if (not arena_product(factors, product)) return false;
    if (product > static_cast<usize>(std::numeric_limits<int>::max()))
    {
        qn::set_err(QNPEPS_ELOC_ERR_OOM);
        result = 0;
        return false;
    }
    result = static_cast<int>(product);
    return true;
}

auto arena_i64_sum(i64 left, i64 right, i64& result) -> bool
{
    if (left < 0 or right < 0 or right > std::numeric_limits<i64>::max() - left)
    {
        qn::set_err(QNPEPS_ELOC_ERR_OOM);
        result = 0;
        return false;
    }
    result = left + right;
    return true;
}

using Carver = qnpeps::ArenaCursor;
using ContextArena = qnpeps::ArenaCursor;

}
