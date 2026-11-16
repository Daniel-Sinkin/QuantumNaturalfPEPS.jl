#ifndef QNPEPS_CONTRACTION_CUH
#define QNPEPS_CONTRACTION_CUH

#include "core/arena_cursor.cuh"
#include "tensor/permutation.cuh"
#include "tensor/tensor.cuh"

#include <vector>

namespace qnpeps
{
class Linalg;

struct ContractTransforms
{
    bool conj_a{};
    bool conj_b{};
};

struct ContractSpec
{
    Shape dims_a{};
    Axes contracted_a{};
    Shape dims_b{};
    Axes contracted_b{};
    ContractTransforms transforms{};
};

struct ContractPlan
{
    Permutation perm_a{};
    Permutation perm_b{};
    Shape result_dim{};
    int m{1};
    int k{1};
    int n{1};
};

[[nodiscard]] auto contract_plan(const ContractSpec& spec) -> ContractPlan;

struct ContractOperand
{
    CuSpanCF32Const src{};
    CuSpanCF32 scratch{};
    cuFloatComplex* const* ptrs{};
};

struct ContractOut
{
    CuSpanCF32 view{};
    cuFloatComplex* const* ptrs{};
};

auto contract(
    ArenaCursor& arena,
    Linalg& la,
    const ContractSpec& spec,
    const DeviceTensor& tensor_a,
    const DeviceTensor& tensor_b,
    DeviceTensor& output
) -> void;

[[nodiscard]] auto contract(
    ArenaCursor& arena,
    Linalg& la,
    const ContractSpec& spec,
    const DeviceTensor& tensor_a,
    const DeviceTensor& tensor_b
) -> DeviceTensor;

auto contract_batched(
    Linalg& la,
    PermutationCache& cache,
    const ContractSpec& spec,
    const ContractOperand& a,
    const ContractOperand& b,
    const ContractOut& out,
    int batch_count
) -> void;

auto contract_strided(
    Linalg& la,
    PermutationCache& cache,
    const ContractSpec& spec,
    const ContractOperand& a,
    const ContractOperand& b,
    const ContractOut& out,
    int batch_count
) -> void;
}

#endif
