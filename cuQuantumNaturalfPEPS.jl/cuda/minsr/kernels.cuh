#ifndef QNPEPS_MINSR_KERNELS_CUH
#define QNPEPS_MINSR_KERNELS_CUH

#include "core/types.cuh"

#include <cuda_runtime.h>

namespace qnpeps
{
class Linalg;
}

namespace qnpeps::minsr
{
using cd = cuDoubleComplex;
using cf = ComplexF32;

struct BetaArgs
{
    cd* output;
    const cf* gram;
    const f64* weights;
    int sample_count;
};

struct BuildMatrixArgs
{
    cd* output;
    const cf* gram;
    const cd* gram_means;
    const f64* weights;
    cd total_mean;
    int sample_count;
};

struct ApplyInverseArgs
{
    cd* values;
    const f64* eigenvalues;
    f64 largest_eigenvalue;
    f64 relative_cut;
    f64 absolute_cut;
    int sample_count;
};

struct ScatterArgs
{
    cd* accumulator;
    const cf* rows;
    i64 row_base;
    i64 compact_count;
    const cd* coefficients;
    const u8* samples;
    const i32* slot_sites;
    int site_count;
    int physical_dimension;
    int row_begin;
    int row_end;
};

__global__ auto cu_canonicalize_eigenvectors(cd* matrix, int order) -> void;
__global__ auto cu_beta(BetaArgs args) -> void;
__global__ auto cu_build_matrix(BuildMatrixArgs args) -> void;
__global__ auto cu_apply_inverse(ApplyInverseArgs args) -> void;
__global__ auto cu_scatter(ScatterArgs args) -> void;
__global__ auto cu_cast_accumulator(cf* output, const cd* accumulator, i64 count) -> void;

[[nodiscard]] auto launch_canonicalize_eigenvectors(Linalg& linalg, cd* matrix, int order)
    -> qnpeps_status;
auto launch_beta(Linalg& linalg, const BetaArgs& args) -> void;
auto launch_build_matrix(Linalg& linalg, const BuildMatrixArgs& args) -> void;
auto launch_apply_inverse(Linalg& linalg, const ApplyInverseArgs& args) -> void;
auto launch_scatter(Linalg& linalg, const ScatterArgs& args) -> void;
auto launch_cast_accumulator(Linalg& linalg, cf* output, const cd* accumulator, i64 count) -> void;
}

#endif
