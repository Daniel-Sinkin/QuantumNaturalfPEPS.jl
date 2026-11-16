#ifndef QNPEPS_ELOC_DENSITY_CUH
#define QNPEPS_ELOC_DENSITY_CUH

#include "core/arena_cursor.cuh"
#include "eloc_fixed.cuh"
#include "linalg/linalg.cuh"

#include <cstddef>
#include <cstdint>

namespace qn_eloc::density
{
struct Context;

struct SiteShape
{
    std::int64_t west{};
    std::int64_t south{};
    std::int64_t east{};
    std::int64_t north{};
};

struct RowArgs
{
    int sites{};
    int max_bond{};
    qnpeps::f64 cutoff{};
    bool contract_up{};
    const SiteShape* shapes{};
    const qn_eloc::fx::cf* const* projected{};
    const qn_eloc::fx::cf* const* adjacent{};
    qn_eloc::fx::cf* const* output{};
    const int* adjacent_ranks{};
    int* output_ranks{};
    qnpeps::f64* device_gauge{};
};

auto create(
    qnpeps::Linalg& linalg,
    qnpeps::ArenaCursor& arena,
    int sites,
    int dim_bond,
    int max_bond,
    Context** output
) -> void;
auto destroy(Context* context) -> void;
auto boundary(Context& context, const RowArgs& args) -> void;
auto contract(Context& context, const RowArgs& args) -> void;
}

template <>
struct std::formatter<qn_eloc::density::SiteShape> : qnpeps::PlainFormatter
{
    auto format(const qn_eloc::density::SiteShape& value, std::format_context& context) const
        -> std::format_context::iterator
    {
        return std::format_to(
            context.out(), "{}x{}x{}x{}", value.west, value.south, value.east, value.north
        );
    }
};

#endif
