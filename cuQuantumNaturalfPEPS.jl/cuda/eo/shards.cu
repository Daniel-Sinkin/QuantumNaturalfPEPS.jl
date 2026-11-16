#include "core/error.hpp"
#include "eo/env_build.cuh"
#include "eo/shards.cuh"

namespace qnpeps::eo
{
auto run_shard(const ShardArgs& args) -> void
{
    qn_eloc_run_impl(
        *args.config,
        args.device_peps,
        args.samples,
        args.n_samples,
        args.terms,
        args.logpsi,
        args.e_loc,
        args.o_rows_device,
        args.o_rows_host,
        nullptr,
        0.0,
        *args.linalg
    );
}

auto gram_tile(const GramTileArgs& args) -> void
{
    if (args.count_a <= 0 or args.count_b <= 0) return;
    qn_eloc_gram_tile_impl(
        *args.config,
        args.rows_a,
        args.samples_a,
        args.count_a,
        args.rows_b,
        args.samples_b,
        args.count_b,
        args.tile,
        args.stream
    );
    check(cudaStreamSynchronize(args.stream));
}
}
