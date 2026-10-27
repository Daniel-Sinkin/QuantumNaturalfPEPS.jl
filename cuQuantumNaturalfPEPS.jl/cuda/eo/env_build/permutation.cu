#include "parts.cuh"

namespace qn_eloc::env
{

auto device_permute(Linalg& linalg, qnpeps::PermutationCache& cache, const PermuteOp& op) -> void
{
    qnpeps::permute_batched(
        cache,
        {
            .dst = {reinterpret_cast<cuFloatComplex*>(op.dst.p), op.dst.stride},
            .src = {reinterpret_cast<const cuFloatComplex*>(op.src.p), op.src.stride},
            .dims_in = qnpeps::Shape{op.dims_in},
            .perm = qnpeps::Permutation{op.perm},
            .batch_count = op.batch,
        },
        linalg.stream()
    );
}

}
