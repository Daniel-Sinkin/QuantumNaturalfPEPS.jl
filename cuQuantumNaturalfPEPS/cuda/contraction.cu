#include "contraction.cuh"
#include "linalg.cuh"

#include <cassert>
#include <limits>
#include <utility>
#include <vector>

namespace qnpeps
{
namespace
{
auto reject_contract() -> bool
{
    set_err(QNPEPS_ERR_INTERNAL);
    return false;
}

[[nodiscard]] auto valid_shape(const Shape& dims) -> bool
{
    if (dims.rank() > static_cast<usize>(k_max_tensor_rank)) return reject_contract();
    for (const auto extent : dims)
        if (extent <= 0) return reject_contract();
    return true;
}

[[nodiscard]] auto valid_axes(const Shape& dims, const Axes& axes) -> bool
{
    bool seen[k_max_tensor_rank]{};
    for (const auto axis : axes)
    {
        if (axis < 0 or axis >= static_cast<int>(dims.rank())) return reject_contract();
        const auto axis_u = static_cast<usize>(axis);
        if (seen[axis_u]) return reject_contract();
        seen[axis_u] = true;
    }
    return true;
}

[[nodiscard]] auto multiply_extent(int& product, int extent) -> bool
{
    if (product > std::numeric_limits<int>::max() / extent) return reject_contract();
    product *= extent;
    return true;
}

[[nodiscard]] auto is_identity(const std::vector<int>& order) -> bool
{
    for (auto i = 0_uz; i < order.size(); ++i)
        if (order[i] != static_cast<int>(i)) return false;
    return true;
}

[[nodiscard]] auto swapped_groups(const std::vector<int>& order, usize first_group_size)
    -> std::vector<int>
{
    std::vector<int> out{};
    out.reserve(order.size());
    for (auto i = first_group_size; i < order.size(); ++i)
        out.push_back(order[i]);
    for (auto i = 0_uz; i < first_group_size; ++i)
        out.push_back(order[i]);
    return out;
}

struct PreparedContract
{
    int m{};
    int k{};
    int n{};
    int b_rows{};
    int b_cols{};
    MatmulConfig op{};
    bool a_materialized{};
    bool b_materialized{};
    bool valid{};
};

[[nodiscard]] auto prepare_contract(
    Linalg& la,
    PermutationCache& cache,
    const ContractSpec& spec,
    const ContractOperand& a,
    const ContractOperand& b,
    int batch_count
) -> PreparedContract
{
    if (batch_count <= 0)
    {
        reject_contract();
        return {};
    }
    const auto plan = contract_plan(spec);
    if (not plan.valid) return {};

    MatmulConfig op{};
    const bool a_materialized{not plan.perm_a.is_identity() or spec.transforms.conj_a};

    bool b_materialized{};
    if (plan.perm_b.is_identity())
    {
        b_materialized = spec.transforms.conj_b;
    }
    else if (is_identity(swapped_groups(plan.perm_b.get(), spec.contracted_b.size())))
    {
        op.op_b = spec.transforms.conj_b ? BlasOp::conj_trans : BlasOp::trans;
    }
    else
    {
        b_materialized = true;
    }

    if ((a_materialized and (not a.src.p or not a.scratch.p))
        or (b_materialized and (not b.src.p or not b.scratch.p)))
    {
        reject_contract();
        return {};
    }

    if (a_materialized)
    {
        permute_batched(
            la,
            cache,
            {
                .dst = a.scratch,
                .src = a.src,
                .dims_in = spec.dims_a,
                .perm = plan.perm_a,
                .batch_count = batch_count,
                .conjugate = spec.transforms.conj_a,
            }
        );
    }

    if (b_materialized)
    {
        permute_batched(
            la,
            cache,
            {
                .dst = b.scratch,
                .src = b.src,
                .dims_in = spec.dims_b,
                .perm = plan.perm_b,
                .batch_count = batch_count,
                .conjugate = spec.transforms.conj_b,
            }
        );
    }

    const bool b_stored_transposed = op.op_b == BlasOp::trans or op.op_b == BlasOp::conj_trans;
    return {
        .m = plan.m,
        .k = plan.k,
        .n = plan.n,
        .b_rows = b_stored_transposed ? plan.n : plan.k,
        .b_cols = b_stored_transposed ? plan.k : plan.n,
        .op = op,
        .a_materialized = a_materialized,
        .b_materialized = b_materialized,
        .valid = true,
    };
}
}

[[nodiscard]] auto contract_plan(const ContractSpec& spec) -> ContractPlan
{
    if (not valid_shape(spec.dims_a) or not valid_shape(spec.dims_b)) return {};
    if (spec.contracted_a.size() != spec.contracted_b.size())
    {
        reject_contract();
        return {};
    }
    if (not valid_axes(spec.dims_a, spec.contracted_a)
        or not valid_axes(spec.dims_b, spec.contracted_b))
        return {};

    for (auto i = 0_uz; i < spec.contracted_a.size(); ++i)
    {
        const auto axis_a = static_cast<usize>(spec.contracted_a[i]);
        const auto axis_b = static_cast<usize>(spec.contracted_b[i]);
        if (spec.dims_a[axis_a] != spec.dims_b[axis_b])
        {
            reject_contract();
            return {};
        }
    }

    const auto free_axes_of = [](const Shape& dims, const Axes& contracted)
    {
        bool is_contracted[k_max_tensor_rank]{};
        for (const auto axis : contracted)
            is_contracted[static_cast<usize>(axis)] = true;
        std::vector<int> free_axes{};
        free_axes.reserve(dims.rank());
        for (auto axis = 0_uz; axis < dims.rank(); ++axis)
            if (not is_contracted[axis]) free_axes.push_back(static_cast<int>(axis));
        return Axes{std::move(free_axes)};
    };
    const auto free_a = free_axes_of(spec.dims_a, spec.contracted_a);
    const auto free_b = free_axes_of(spec.dims_b, spec.contracted_b);

    std::vector<int> perm_a{free_a.begin(), free_a.end()};
    perm_a.insert(perm_a.end(), spec.contracted_a.begin(), spec.contracted_a.end());
    std::vector<int> perm_b{spec.contracted_b.begin(), spec.contracted_b.end()};
    perm_b.insert(perm_b.end(), free_b.begin(), free_b.end());

    int m{1};
    int k{1};
    int n{1};
    std::vector<int> result_dim{};
    for (const auto axis : free_a)
    {
        const auto extent = spec.dims_a[static_cast<usize>(axis)];
        if (not multiply_extent(m, extent)) return {};
        result_dim.push_back(extent);
    }
    for (const auto axis : free_b)
    {
        const auto extent = spec.dims_b[static_cast<usize>(axis)];
        if (not multiply_extent(n, extent)) return {};
        result_dim.push_back(extent);
    }
    for (const auto axis : spec.contracted_a)
        if (not multiply_extent(k, spec.dims_a[static_cast<usize>(axis)])) return {};
    if (result_dim.empty()) result_dim.push_back(1);

    ContractPlan plan{
        .perm_a = Permutation{std::move(perm_a)},
        .perm_b = Permutation{std::move(perm_b)},
        .result_dim = Shape{std::move(result_dim)},
        .m = m,
        .k = k,
        .n = n,
        .valid = true,
    };
    assert(plan.perm_a.can_apply_to(spec.dims_a));
    assert(plan.perm_b.can_apply_to(spec.dims_b));
    assert(plan.result_dim.num_elems() == static_cast<usize>(m) * static_cast<usize>(n));
    return plan;
}

auto contract(
    ArenaCursor& arena,
    Linalg& la,
    const ContractSpec& spec,
    const DeviceTensor& tensor_a,
    const DeviceTensor& tensor_b
) -> DeviceTensor
{
    const auto plan = contract_plan(spec);
    if (not plan.valid) return {};
    if (spec.dims_a != tensor_a.dim or spec.dims_b != tensor_b.dim)
    {
        reject_contract();
        return {};
    }
    if (not tensor_a.d or not tensor_b.d)
    {
        reject_contract();
        return {};
    }

    auto result = alloc(arena, plan.result_dim);
    if (not result.d) return result;

    ArenaCursor scratch_frame{arena};
    const auto m = static_cast<usize>(plan.m);
    const auto k = static_cast<usize>(plan.k);
    const auto n = static_cast<usize>(plan.n);
    const auto a_perm_bytes = device_align(m * k * sizeof(cuFloatComplex));
    const auto b_perm_bytes = device_align(k * n * sizeof(cuFloatComplex));
    auto* scratch = scratch_frame.take<char>(a_perm_bytes + b_perm_bytes);
    if (not scratch) return result;

    auto* a_perm = byte_offset<cuFloatComplex>(scratch, 0);
    auto* b_perm = byte_offset<cuFloatComplex>(scratch, a_perm_bytes);
    permute_axes(tensor_a, plan.perm_a, spec.transforms.conj_a, a_perm);
    permute_axes(tensor_b, plan.perm_b, spec.transforms.conj_b, b_perm);
    if (err_state() != QNPEPS_OK) return result;

    la.matmul(
        CuMatrixConst{cf_cast(a_perm), plan.m, plan.k},
        CuMatrixConst{cf_cast(b_perm), plan.k, plan.n},
        CuMatrix{cf_cast(result.d), plan.m, plan.n}
    );
    return result;
}

auto contract_batched(
    Linalg& la,
    PermutationCache& cache,
    const ContractSpec& spec,
    const ContractOperand& a,
    const ContractOperand& b,
    const ContractOut& out,
    int batch_count
) -> void
{
    if (batch_count <= 0)
    {
        reject_contract();
        return;
    }
    if (not a.ptrs or not b.ptrs or not out.ptrs)
    {
        reject_contract();
        return;
    }
    const auto prepared = prepare_contract(la, cache, spec, a, b, batch_count);
    if (not prepared.valid or err_state() != QNPEPS_OK) return;
    la.matmul_batched_ptr(
        a.ptrs,
        prepared.m,
        prepared.k,
        b.ptrs,
        prepared.b_rows,
        prepared.b_cols,
        out.ptrs,
        prepared.m,
        prepared.n,
        batch_count,
        prepared.op
    );
}

auto contract_strided(
    Linalg& la,
    PermutationCache& cache,
    const ContractSpec& spec,
    const ContractOperand& a,
    const ContractOperand& b,
    const ContractOut& out,
    int batch_count
) -> void
{
    if (batch_count <= 0)
    {
        reject_contract();
        return;
    }
    if (not a.src.p or not b.src.p or not out.view.p)
    {
        reject_contract();
        return;
    }
    const auto prepared = prepare_contract(la, cache, spec, a, b, batch_count);
    if (not prepared.valid or err_state() != QNPEPS_OK) return;
    const CuArrayConst a_view = prepared.a_materialized ? CuArrayConst{a.scratch} : a.src;
    const CuArrayConst b_view = prepared.b_materialized ? CuArrayConst{b.scratch} : b.src;
    la.matmul_batched(
        {a_view, prepared.m, prepared.k},
        {b_view, prepared.b_rows, prepared.b_cols},
        {out.view, prepared.m, prepared.n},
        batch_count,
        prepared.op
    );
}
}
