#ifndef QNPEPS_MINSR_CUH
#define QNPEPS_MINSR_CUH

#include "capi/qnpeps.h"
#include "core/types.cuh"

namespace qnpeps::minsr
{
inline constexpr i64 k_host_tile_bytes_default{64 * 1024 * 1024};

[[nodiscard]] auto descriptor_dense_count(const QnpepsMinsrDesc* descriptor) -> i64;
[[nodiscard]] auto descriptor_compact_count(const QnpepsMinsrDesc* descriptor) -> i64;
[[nodiscard]] auto descriptor_scratch_bytes(const QnpepsMinsrDesc* descriptor) -> i64;

auto execute(const QnpepsMinsrDesc* descriptor, const QnpepsMinsrArgs* args) -> void;
auto ctx_create(const QnpepsMinsrDesc* descriptor, void* stream, qnpeps_minsr_ctx** out) -> void;
auto ctx_run(qnpeps_minsr_ctx* ctx, const QnpepsMinsrArgs* args) -> void;
auto ctx_destroy(qnpeps_minsr_ctx* ctx) -> void;
}

#endif
