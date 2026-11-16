#include "capi/internal/validation.cuh"
#include "capi/qnpeps.h"
#include "dlenv/build.cuh"
#include "zipup/zipup_mpo_mps.cuh"

#include <cstdint>

using namespace qnpeps;
using qnpeps::capi::validate_config;

extern "C" int64_t qnpeps_zipup_peps_row_bytes(const QnpepsConfig* config, int maxdim)
{
    return guarded_or(
        int64_t{-1},
        [&]
        {
            validate_config(config);
            return qnpeps::dlenv::zipup_peps_row_bytes(*config, maxdim);
        }
    );
}

extern "C" qnpeps_status qnpeps_zipup_ctx_create(
    const QnpepsConfig* config, int maxdim, void* stream, qnpeps_zipup_ctx** out
)
{
    return guarded(
        [&]
        {
            require(out != nullptr, Status::null_argument, "zipup context output is null");
            *out = nullptr;
            validate_config(config);
            require(
                maxdim >= 1, Status::bad_config, "zipup context maxdim {} is not positive", maxdim
            );
            *out = qnpeps::dlenv::create_zipup_context(
                *config, maxdim, static_cast<cudaStream_t>(stream)
            );
        }
    );
}

extern "C" void qnpeps_zipup_ctx_destroy(qnpeps_zipup_ctx* context)
{
    static_cast<void>(guarded_keeping_message([&]
                                              { qnpeps::dlenv::destroy_zipup_context(context); }));
}

extern "C" qnpeps_status qnpeps_zipup_ctx_begin(qnpeps_zipup_ctx* context)
{
    return guarded(
        [&]
        {
            require(context != nullptr, Status::null_argument, "zipup begin without a context");
            qnpeps::dlenv::begin_zipup_context(*context);
        }
    );
}

extern "C" qnpeps_status qnpeps_zipup_ctx_enqueue_peps_row(
    qnpeps_zipup_ctx* context, const QnpepsZipupPepsRowArgs* args
)
{
    return guarded(
        [&]
        {
            require(
                context != nullptr and args != nullptr,
                Status::null_argument,
                "zipup row needs a context and arguments"
            );
            require_struct_size(*args, "zipup row arguments");
            const auto has_buffers = args->peps_row and args->output_dims and args->output_values;
            require(has_buffers, Status::null_argument, "zipup row misses a buffer");
            qnpeps::dlenv::enqueue_peps_row(*context, *args);
        }
    );
}

extern "C" qnpeps_status qnpeps_zipup_ctx_finish(
    qnpeps_zipup_ctx* context, double* scales, uint64_t count
)
{
    return guarded(
        [&]
        {
            require(
                context != nullptr and scales != nullptr,
                Status::null_argument,
                "zipup finish needs a context and a scale output"
            );
            qnpeps::dlenv::finish_zipup_context(*context, scales, static_cast<usize>(count));
        }
    );
}

extern "C" int64_t qnpeps_zipup_mpo_mps_bytes(const QnpepsZipupMpoMpsDesc* descriptor)
{
    return guarded_or(int64_t{-1}, [&] { return qnpeps::zipup::output_bytes(descriptor); });
}

extern "C" qnpeps_status qnpeps_zipup_mpo_mps(
    const QnpepsZipupMpoMpsDesc* descriptor, const QnpepsZipupMpoMpsArgs* args
)
{
    return guarded([&] { qnpeps::zipup::execute(descriptor, args); });
}
