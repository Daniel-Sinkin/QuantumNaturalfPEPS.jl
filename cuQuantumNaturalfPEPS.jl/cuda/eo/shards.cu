#include "core/error.cuh"
#include "eo/env_build.cuh"
#include "eo/shards.cuh"

namespace qnpeps::eo
{
auto run_shard(const ShardArgs& args) -> qnpeps_status
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
    return qnpeps::err_state();
}
}
