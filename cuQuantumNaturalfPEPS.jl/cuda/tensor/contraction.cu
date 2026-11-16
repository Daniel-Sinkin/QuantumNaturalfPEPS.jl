#include "linalg/linalg.cuh"
#include "tensor/contraction.cuh"

#include <cassert>
#include <limits>
#include <utility>
#include <vector>

namespace qnpeps
{
namespace
{
auto multiply_extent(int& product, int extent) -> void
{
    require(
        product <= std::numeric_limits<int>::max() / extent,
        "contraction extent {} times {} exceeds the int range",
        product,
        extent
    );
    product *= extent;
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

[[nodiscard]] auto free_axes_of(const Shape& dims, const Axes& contracted) -> Axes
{
    qnpeps::CuArray<bool, k_max_tensor_rank> is_contracted{};
    for (const auto axis : contracted)
        is_contracted[static_cast<usize>(axis)] = true;

    std::vector<int> free_axes{};
    free_axes.reserve(dims.rank());
    for (auto axis = 0_uz; axis < dims.rank(); ++axis)
        if (not is_contracted[axis]) free_axes.push_back(static_cast<int>(axis));
    return Axes{std::move(free_axes)};
}

struct PreparedContraction
{
    int m{};
    int k{};
    int n{};
    int b_rows{};
    int b_cols{};
    MatmulConfig op{};
    bool a_materialized{};
    bool b_materialized{};
};

[[nodiscard]] auto prepare_contraction(
    Linalg& la,
    PermutationCache& cache,
    const ContractSpec& spec,
    const ContractOperand& a,
    const ContractOperand& b,
    int batch_count
) -> PreparedContraction
{
    require(batch_count > 0, "contraction over {} lanes", batch_count);
    const auto plan = contract_plan(spec);

    MatmulConfig op{};
    const auto a_materialized = not plan.perm_a.is_identity() or spec.transforms.conj_a;

    auto b_materialized = false;
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

    const auto materialize = [&](const ContractOperand& operand,
                                 const Shape& dims,
                                 const Permutation& permutation,
                                 bool conjugate)
    {
        require(
            operand.src.p != nullptr and operand.scratch.p != nullptr,
            "contraction operand needs a source and a scratch buffer to permute"
        );
        permute_batched(
            cache,
            {.dst = operand.scratch,
             .src = operand.src,
             .dims_in = dims,
             .perm = permutation,
             .batch_count = batch_count,
             .conjugate = conjugate},
            la.stream()
        );
    };

    if (a_materialized) materialize(a, spec.dims_a, plan.perm_a, spec.transforms.conj_a);
    if (b_materialized) materialize(b, spec.dims_b, plan.perm_b, spec.transforms.conj_b);

    auto b_rows = plan.k;
    auto b_cols = plan.n;
    if (is_trans(op.op_b)) std::swap(b_rows, b_cols);
    return PreparedContraction{
        .m = plan.m,
        .k = plan.k,
        .n = plan.n,
        .b_rows = b_rows,
        .b_cols = b_cols,
        .op = op,
        .a_materialized = a_materialized,
        .b_materialized = b_materialized,
    };
}
}

[[nodiscard]] auto contract_plan(const ContractSpec& spec) -> ContractPlan
{
    const auto& axes_a = spec.contracted_a;
    const auto& dims_a = spec.dims_a;
    const auto& axes_b = spec.contracted_b;
    const auto& dims_b = spec.dims_b;
    require(
        axes_a.size() == axes_b.size(),
        "contraction pairs {} axes with {} axes",
        axes_a.size(),
        axes_b.size()
    );
    const auto axes_apply = axes_a.can_apply_to(dims_a) and axes_b.can_apply_to(dims_b);
    require(
        axes_apply,
        "contracted axes exceed the operand ranks {} and {}",
        dims_a.rank(),
        dims_b.rank()
    );

    for (auto i = 0_uz; i < axes_a.size(); ++i)
    {
        const auto axis_a = static_cast<usize>(axes_a[i]);
        const auto axis_b = static_cast<usize>(axes_b[i]);
        require(
            dims_a[axis_a] == dims_b[axis_b],
            "contracted axis {} has extent {}, its partner axis {} has extent {}",
            axis_a,
            dims_a[axis_a],
            axis_b,
            dims_b[axis_b]
        );
    }

    const auto free_a = free_axes_of(dims_a, axes_a);
    const auto free_b = free_axes_of(dims_b, axes_b);

    std::vector<int> order_a{free_a.begin(), free_a.end()};
    order_a.insert(order_a.end(), axes_a.begin(), axes_a.end());
    std::vector<int> order_b{axes_b.begin(), axes_b.end()};
    order_b.insert(order_b.end(), free_b.begin(), free_b.end());

    int m{1};
    int k{1};
    int n{1};
    std::vector<int> result_dim{};
    for (const auto axis : free_a)
    {
        const auto extent = dims_a[static_cast<usize>(axis)];
        multiply_extent(m, extent);
        result_dim.push_back(extent);
    }
    for (const auto axis : free_b)
    {
        const auto extent = dims_b[static_cast<usize>(axis)];
        multiply_extent(n, extent);
        result_dim.push_back(extent);
    }
    for (const auto axis : axes_a)
        multiply_extent(k, dims_a[static_cast<usize>(axis)]);
    if (result_dim.empty()) result_dim.push_back(1);

    ContractPlan plan{
        .perm_a = Permutation{std::move(order_a)},
        .perm_b = Permutation{std::move(order_b)},
        .result_dim = Shape{std::move(result_dim)},
        .m = m,
        .k = k,
        .n = n,
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
    const DeviceTensor& tensor_b,
    DeviceTensor& output
) -> void
{
    output = {};
    const auto plan = contract_plan(spec);
    const auto shapes_match = spec.dims_a == tensor_a.dim and spec.dims_b == tensor_b.dim;
    require(
        shapes_match,
        "contraction spec ranks {} and {} disagree with tensor ranks {} and {}",
        spec.dims_a.rank(),
        spec.dims_b.rank(),
        tensor_a.dim.rank(),
        tensor_b.dim.rank()
    );
    require(
        tensor_a.d != nullptr and tensor_b.d != nullptr, "contraction of an unallocated tensor"
    );

    output = alloc(arena, plan.result_dim);

    ArenaCursor scratch_frame{arena};

    const auto m = static_cast<usize>(plan.m);
    const auto k = static_cast<usize>(plan.k);
    const auto n = static_cast<usize>(plan.n);

    const auto a_perm_bytes = device_align(m * k * sizeof(cuFloatComplex));
    const auto b_perm_bytes = device_align(k * n * sizeof(cuFloatComplex));

    auto* scratch = scratch_frame.take<char>(a_perm_bytes + b_perm_bytes);

    auto* a_perm = byte_offset<cuFloatComplex>(scratch, 0);
    auto* b_perm = byte_offset<cuFloatComplex>(scratch, a_perm_bytes);
    permute_axes(tensor_a, plan.perm_a, spec.transforms.conj_a, a_perm, la.stream());
    permute_axes(tensor_b, plan.perm_b, spec.transforms.conj_b, b_perm, la.stream());

    la.matmul(
        CuMatrixCF32Const{a_perm, plan.m, plan.k},
        CuMatrixCF32Const{b_perm, plan.k, plan.n},
        CuMatrixCF32{output.d, plan.m, plan.n}
    );
}

[[nodiscard]] auto contract(
    ArenaCursor& arena,
    Linalg& la,
    const ContractSpec& spec,
    const DeviceTensor& tensor_a,
    const DeviceTensor& tensor_b
) -> DeviceTensor
{
    DeviceTensor output{};
    contract(arena, la, spec, tensor_a, tensor_b, output);
    return output;
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
    const auto has_pointer_arrays = a.ptrs and b.ptrs and out.ptrs;
    require(
        has_pointer_arrays,
        "pointer-batched contraction over {} lanes misses a pointer array",
        batch_count
    );

    const auto prepared = prepare_contraction(la, cache, spec, a, b, batch_count);

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
    const auto has_buffers = a.src.p and b.src.p and out.view.p;
    require(has_buffers, "strided contraction over {} lanes misses an operand buffer", batch_count);
    const auto prepared = prepare_contraction(la, cache, spec, a, b, batch_count);
    const auto a_view = prepared.a_materialized ? CuSpanCF32Const{a.scratch} : a.src;
    const auto b_view = prepared.b_materialized ? CuSpanCF32Const{b.scratch} : b.src;
    la.matmul_batched(
        {a_view, prepared.m, prepared.k},
        {b_view, prepared.b_rows, prepared.b_cols},
        {out.view, prepared.m, prepared.n},
        batch_count,
        prepared.op
    );
}
}
