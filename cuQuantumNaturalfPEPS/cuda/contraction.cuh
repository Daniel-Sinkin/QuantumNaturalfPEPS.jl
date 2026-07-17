#ifndef QNPEPS_CONTRACTION_CUH
#define QNPEPS_CONTRACTION_CUH

#include "arena_cursor.cuh"
#include "dtensor.cuh"

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
    bool valid{};
};

[[nodiscard]] auto contract_plan(const ContractSpec& spec) -> ContractPlan;

struct ContractOperand
{
    CuArrayConst src{};
    CuArray scratch{};
    cf* const* ptrs{};
};

struct ContractOut
{
    CuArray view{};
    cf* const* ptrs{};
};

auto contract(
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
