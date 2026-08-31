#ifndef QNPEPS_GRAM_CUH
#define QNPEPS_GRAM_CUH

#include "capi/qnpeps.h"
#include "core/types.cuh"

namespace qnpeps
{
class Linalg;
}

namespace qnpeps::gram
{
auto ctx_create(const QnpepsGramDesc* descriptor, void* stream, qnpeps_gram_ctx** out)
    -> qnpeps_status;
auto ctx_create(const QnpepsGramDesc* descriptor, Linalg& linalg, qnpeps_gram_ctx** out)
    -> qnpeps_status;
auto ctx_run(qnpeps_gram_ctx* ctx, const QnpepsGramArgs* args) -> qnpeps_status;
auto ctx_footprint(const qnpeps_gram_ctx* ctx, QnpepsGramFootprint* out) -> qnpeps_status;
auto ctx_destroy(qnpeps_gram_ctx* ctx) -> void;
}

#endif
