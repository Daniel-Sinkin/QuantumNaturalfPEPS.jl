#ifndef QNPEPS_DLENV_BUILD_ZIPUP_CONTEXT_CUH
#define QNPEPS_DLENV_BUILD_ZIPUP_CONTEXT_CUH

#include "dlenv/build/arena.cuh"
#include "dlenv/build/rows.cuh"

namespace qnpeps::dlenv
{

[[nodiscard]] auto zipup_peps_row_bytes(const QnpepsConfig& config, int maxdim) -> i64;
auto create_zipup_context(const QnpepsConfig& config, int maxdim, cudaStream_t stream)
    -> qnpeps_zipup_ctx*;
auto destroy_zipup_context(qnpeps_zipup_ctx* context) -> void;
auto begin_zipup_context(qnpeps_zipup_ctx& context) -> void;
auto enqueue_peps_row(qnpeps_zipup_ctx& context, const QnpepsZipupPepsRowArgs& args) -> void;
auto finish_zipup_context(qnpeps_zipup_ctx& context, f64* scales, usize count) -> void;

}

#endif
