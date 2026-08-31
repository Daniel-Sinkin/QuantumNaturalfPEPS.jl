#include "../gram/gram.cuh"
#include "common.cuh"

#include <mutex>
#include <new>

namespace
{

auto e2e_status(qnpeps_status status) -> qnpeps_e2e_status
{
    return static_cast<qnpeps_e2e_status>(status);
}

auto check_config(const QnpepsE2eConfig* config) -> qnpeps_e2e_status
{
    if (not config) return QNPEPS_E2E_ERR_NULL_ARG;
    if (config->struct_size != sizeof(QnpepsE2eConfig)) return QNPEPS_E2E_ERR_BAD_VERSION;
    if (config->lx < 2 or config->ly < 2 or config->dim_phys != 2 or config->dim_bond < 1)
        return QNPEPS_E2E_ERR_BAD_CONFIG;
    return QNPEPS_E2E_OK;
}

}

struct qnpeps_e2e_gram_ctx
{
    qnpeps_gram_ctx* root{};
    int device{};
    QnpepsGramFootprint root_footprint{};
    QnpepsE2eGramFootprint footprint{};
    std::mutex mutex{};
};

extern "C" qnpeps_e2e_status qnpeps_e2e_gram_ctx_create(
    const QnpepsE2eConfig* config, int64_t n_samples, void* stream, qnpeps_e2e_gram_ctx** out
)
{
    qnpeps::reset_err();
    if (not out) return QNPEPS_E2E_ERR_NULL_ARG;
    *out = nullptr;
    const auto config_status = check_config(config);
    if (config_status != QNPEPS_E2E_OK) return config_status;
    if (n_samples < 2) return QNPEPS_E2E_ERR_BAD_CONFIG;

    auto* context = new (std::nothrow) qnpeps_e2e_gram_ctx;
    if (not context) return QNPEPS_E2E_ERR_OOM;
    qnpeps_status status{qnpeps::cuda_status(cudaGetDevice(&context->device))};
    const QnpepsGramDesc descriptor{
        .struct_size = sizeof(QnpepsGramDesc),
        .lx = config->lx,
        .ly = config->ly,
        .dim_phys = config->dim_phys,
        .dim_bond = config->dim_bond,
        .consumer = QNPEPS_GRAM_CONSUMER_SLAB,
        .reserved = 0,
        .n_samples = n_samples
    };
    if (status == QNPEPS_OK) status = qnpeps::gram::ctx_create(&descriptor, stream, &context->root);
    if (status == QNPEPS_OK)
    {
        context->root_footprint.struct_size = sizeof(QnpepsGramFootprint);
        status = qnpeps::gram::ctx_footprint(context->root, &context->root_footprint);
    }
    if (status == QNPEPS_OK)
    {
        const auto& root = context->root_footprint;
        context->footprint = {
            .struct_size = sizeof(QnpepsE2eGramFootprint),
            .reserved = 0,
            .context_device_bytes =
                root.geometry_device_bytes + root.dense_a_device_bytes + root.dense_b_device_bytes,
            .geometry_device_bytes = root.geometry_device_bytes,
            .dense_a_device_bytes = root.dense_a_device_bytes,
            .dense_b_device_bytes = root.dense_b_device_bytes,
            .caller_samples_bytes = root.caller_samples_bytes,
            .caller_rows_bytes = root.caller_rows_bytes,
            .caller_gram_bytes = root.caller_gram_bytes
        };
    }
    if (status != QNPEPS_OK)
    {
        qnpeps::gram::ctx_destroy(context->root);
        delete context;
        return e2e_status(status);
    }
    *out = context;
    return QNPEPS_E2E_OK;
}

extern "C" qnpeps_e2e_status qnpeps_e2e_gram_ctx_run(
    qnpeps_e2e_gram_ctx* context,
    const uint8_t* device_samples,
    const qnpeps_e2e_cbuf* o_rows_device,
    qnpeps_e2e_cbuf* raw_gram_device,
    QnpepsE2eGramTimings* timings_out
)
{
    qnpeps::reset_err();
    if (not context or not device_samples or not o_rows_device or not raw_gram_device)
        return QNPEPS_E2E_ERR_NULL_ARG;
    if (timings_out and timings_out->struct_size != sizeof(QnpepsE2eGramTimings))
        return QNPEPS_E2E_ERR_BAD_VERSION;
    const std::lock_guard<std::mutex> lock{context->mutex};

    int caller{};
    qnpeps_status status{qnpeps::cuda_status(cudaGetDevice(&caller))};
    const bool have_caller{status == QNPEPS_OK};
    if (status == QNPEPS_OK) status = qnpeps::cuda_status(cudaSetDevice(context->device));
    const QnpepsGramArgs args{
        .struct_size = sizeof(QnpepsGramArgs),
        .reserved = 0,
        .samples = device_samples,
        .samples_bytes = context->root_footprint.caller_samples_bytes,
        .o_rows = o_rows_device,
        .o_rows_bytes = context->root_footprint.caller_rows_bytes,
        .gram_out = raw_gram_device,
        .gram_out_bytes = context->root_footprint.caller_gram_bytes,
        .stream = nullptr
    };
    if (status == QNPEPS_OK) status = qnpeps::gram::ctx_run(context->root, &args);
    const auto restore = have_caller ? cudaSetDevice(caller) : cudaSuccess;
    if (status == QNPEPS_OK) status = qnpeps::cuda_status(restore);

    if (timings_out)
    {
        const uint32_t struct_size{timings_out->struct_size};
        *timings_out = {};
        timings_out->struct_size = struct_size;
    }
    return e2e_status(status);
}

extern "C" qnpeps_e2e_status qnpeps_e2e_gram_ctx_footprint(
    const qnpeps_e2e_gram_ctx* context, QnpepsE2eGramFootprint* out
)
{
    if (not context or not out) return QNPEPS_E2E_ERR_NULL_ARG;
    if (out->struct_size != sizeof(QnpepsE2eGramFootprint)) return QNPEPS_E2E_ERR_BAD_VERSION;
    *out = context->footprint;
    return QNPEPS_E2E_OK;
}

extern "C" void qnpeps_e2e_gram_ctx_destroy(qnpeps_e2e_gram_ctx* context)
{
    if (not context) return;
    qnpeps::gram::ctx_destroy(context->root);
    delete context;
}
