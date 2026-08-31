#include "../minsr/solve.cuh"
#include "capi/qnpeps.h"
#include "common.cuh"
#include "core/arena_cursor.cuh"
#include "core/session.cuh"
#include "dans_qnpeps_e2e.h"
#include "dans_qnpeps_eloc.h"
#include "dlenv/build.cuh"
#include "eo/env_build.cuh"
#include "layout.cuh"
#include "sampler/draw.cuh"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace
{

using qn_e2e::cf;
using qn_e2e::err_state;
using qn_e2e::set_err;

auto sampler_cfg(const QnpepsE2eConfig& c) -> QnpepsConfig
{
    QnpepsConfig s{};
    s.struct_size = sizeof(QnpepsConfig);
    s.lx = c.lx;
    s.ly = c.ly;
    s.dim_phys = c.dim_phys;
    s.dim_bond = c.dim_bond;
    s.chi_s = c.chi_s;
    s.chi_dl = c.chi_dl;
    s.seed = c.seed;
    s.sampling_mode = c.sampling_mode;
    s.chi_c = c.contract_dim;
    return s;
}

auto sample_batch(const QnpepsE2eConfig& c, i64 ns) -> i64
{
    return c.sample_batch > 0 ? c.sample_batch : std::min<i64>(ns, qnpeps::k_max_batch_size);
}

auto eloc_cfg(const QnpepsE2eConfig& c) -> QnpepsElocConfig
{
    QnpepsElocConfig e{};
    e.struct_size = sizeof(QnpepsElocConfig);
    e.lx = c.lx;
    e.ly = c.ly;
    e.dim_phys = c.dim_phys;
    e.dim_bond = c.dim_bond;
    e.chi_eo = c.chi_eo;
    e.meo = c.meo;
    return e;
}

auto map_sampler(qnpeps_status s) -> void
{
    if (s == QNPEPS_OK) return;
    if (s == QNPEPS_ERR_BAD_CONFIG)
        set_err(QNPEPS_E2E_ERR_BAD_CONFIG);
    else if (s == QNPEPS_ERR_CUDA)
        set_err(QNPEPS_E2E_ERR_CUDA);
    else if (s == QNPEPS_ERR_OOM)
        set_err(QNPEPS_E2E_ERR_OOM);
    else
        set_err(QNPEPS_E2E_ERR_INTERNAL);
}

template <class T>
auto dev_alloc(i64 count) -> T*
{
    if (err_state() != QNPEPS_E2E_OK) return nullptr;
    void* p{};
    const cudaError_t e{cudaMalloc(&p, sizeof(T) * static_cast<usize>(count < 1 ? 1 : count))};
    if (e != cudaSuccess)
    {
        set_err(QNPEPS_E2E_ERR_OOM);
        return nullptr;
    }
    return static_cast<T*>(p);
}

auto d2d(void* dst, const void* src, usize bytes, cudaStream_t stream) -> void
{
    if (not dst) return;
    QN_E2E_CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
}

auto check_cfg(const QnpepsE2eConfig* c) -> qnpeps_e2e_status
{
    if (not c) return QNPEPS_E2E_ERR_NULL_ARG;
    if (c->struct_size != sizeof(QnpepsE2eConfig)) return QNPEPS_E2E_ERR_BAD_VERSION;
    if (c->lx < 2 or c->ly < 2 or c->dim_phys != 2 or c->dim_bond < 1)
        return QNPEPS_E2E_ERR_BAD_CONFIG;
    if (c->chi_s < 1 or c->chi_dl < 1 or c->chi_eo < 1 or c->meo < 1)
        return QNPEPS_E2E_ERR_BAD_CONFIG;
    if (c->sampling_mode != QNPEPS_SAMPLING_FAST and c->sampling_mode != QNPEPS_SAMPLING_FULL)
        return QNPEPS_E2E_ERR_BAD_CONFIG;
    if (c->sampling_mode == QNPEPS_SAMPLING_FULL and c->contract_dim < 1)
        return QNPEPS_E2E_ERR_BAD_CONFIG;
    if (c->sample_batch < 0 or c->sample_batch > qnpeps::k_max_batch_size)
        return QNPEPS_E2E_ERR_BAD_CONFIG;
    return QNPEPS_E2E_OK;
}

}

namespace qn_e2e
{

auto step_single(
    const QnpepsE2eConfig& cfg,
    const void* device_peps,
    i64 ns,
    const void* terms,
    i64 host_tile_bytes,
    f64 relative_cut,
    f64 absolute_cut,
    cf* theta_dot_out,
    f64* e_mean_out,
    f64* e_var_out,
    f64* ess_out,
    u8* samples_out,
    f64* logq_out,
    f64* log_gauge_out,
    f64* logpsi_out,
    f64* e_loc_out,
    cf* o_rows_host_out,
    qnpeps::Linalg& linalg
) -> void
{
    const auto s = linalg.stream();
    const QnpepsConfig scfg{sampler_cfg(cfg)};
    const QnpepsElocConfig lcfg{eloc_cfg(cfg)};
    const i64 sites{static_cast<i64>(cfg.lx) * cfg.ly};
    const i64 compact{qn_e2e::compact_count(cfg)};
    const i64 dim_batch{sample_batch(cfg, ns)};

    const i64 dlenv_bytes{qnpeps_dlenv_bytes(&scfg)};
    const i64 samples_bytes{qnpeps_sample_bytes(&scfg, static_cast<u64>(ns))};
    if (dlenv_bytes < 0 or samples_bytes < 0)
    {
        set_err(QNPEPS_E2E_ERR_BAD_CONFIG);
        return;
    }

    auto dlenv{dev_alloc<u8>(dlenv_bytes)};
    auto rowlogs{dev_alloc<f64>(cfg.lx - 1)};
    auto samples{dev_alloc<u8>(samples_bytes)};
    auto logq{dev_alloc<f64>(ns)};
    auto loggauge{dev_alloc<f64>(ns)};
    auto logpsi{dev_alloc<f64>(2 * ns)};
    auto eloc{dev_alloc<f64>(2 * ns)};
    auto rows{dev_alloc<cf>(ns * compact)};
    auto gram{dev_alloc<cf>(ns * ns)};

    if (err_state() == QNPEPS_E2E_OK)
    {
        map_sampler(qnpeps::dlenv::build_dlenv_packed(scfg, device_peps, dlenv, rowlogs, linalg));
    }
    if (err_state() == QNPEPS_E2E_OK)
    {
        const qnpeps::sampler::SampleArgs sample_args{
            .device_peps = device_peps,
            .device_dlenv = dlenv,
            .scratch = nullptr,
            .scratch_bytes = 0,
            .output = samples,
            .logpc_out = logq,
            .lognorm_out = loggauge,
            .n_samples = static_cast<u64>(ns),
            .batch_base = 0,
            .dim_batch = static_cast<u64>(dim_batch),
            .stream = s,
            .output_location = qnpeps::sampler::SampleOutputLocation::device
        };
        map_sampler(qnpeps::sampler::sample(scfg, sample_args, linalg));
    }
    if (err_state() == QNPEPS_E2E_OK)
    {
        qn_eloc_run_impl(
            lcfg,
            reinterpret_cast<const cf*>(device_peps),
            samples,
            ns,
            static_cast<const QnpepsElocTermTable*>(terms),
            logpsi,
            eloc,
            rows,
            o_rows_host_out,
            gram,
            0.0,
            linalg
        );
        QN_E2E_CUDA_CHECK(cudaStreamSynchronize(s));
    }
    if (err_state() == QNPEPS_E2E_OK)
    {
        qnpeps::minsr::qn_e2e_minsr_impl(
            cfg,
            ns,
            samples,
            logpsi,
            eloc,
            logq,
            gram,
            rows,
            nullptr,
            host_tile_bytes,
            relative_cut,
            absolute_cut,
            theta_dot_out,
            e_mean_out,
            e_var_out,
            ess_out,
            linalg
        );
    }
    if (err_state() == QNPEPS_E2E_OK)
    {
        const auto sample_count = static_cast<usize>(ns);
        const auto scalar_count = 2 * sample_count;
        d2d(samples_out, samples, sample_count * static_cast<usize>(sites), s);
        d2d(logq_out, logq, sizeof(f64) * sample_count, s);
        d2d(log_gauge_out, loggauge, sizeof(f64) * sample_count, s);
        d2d(logpsi_out, logpsi, sizeof(f64) * scalar_count, s);
        d2d(e_loc_out, eloc, sizeof(f64) * scalar_count, s);
        QN_E2E_CUDA_CHECK(cudaStreamSynchronize(s));
    }

    cudaFree(dlenv);
    cudaFree(rowlogs);
    cudaFree(samples);
    cudaFree(logq);
    cudaFree(loggauge);
    cudaFree(logpsi);
    cudaFree(eloc);
    cudaFree(rows);
    cudaFree(gram);
}

}

extern "C" qnpeps_e2e_status qnpeps_e2e_step(
    const QnpepsE2eConfig* cfg,
    const void* device_peps,
    int64_t n_samples,
    const void* terms,
    int64_t host_tile_bytes,
    double relative_cut,
    double absolute_cut,
    qnpeps_e2e_cbuf* theta_dot_out,
    double* e_mean_out,
    double* e_var_out,
    double* ess_out,
    uint8_t* samples_out,
    double* logq_out,
    double* log_gauge_out,
    double* logpsi_out,
    double* e_loc_out,
    qnpeps_e2e_cbuf* o_rows_host,
    void* stream
)
{
    const qnpeps_e2e_status v{check_cfg(cfg)};
    if (v != QNPEPS_E2E_OK) return v;
    const auto missing_output = not device_peps or not terms or not theta_dot_out or not e_mean_out
                                or not e_var_out or not ess_out;
    if (missing_output) return QNPEPS_E2E_ERR_NULL_ARG;
    if (n_samples < 2) return QNPEPS_E2E_ERR_BAD_CONFIG;

    qn_e2e::clear_err();
    auto session = qnpeps::make_session(static_cast<cudaStream_t>(stream));
    if (not session) return err_state();
    qn_e2e::step_single(
        *cfg,
        device_peps,
        n_samples,
        terms,
        host_tile_bytes,
        relative_cut,
        absolute_cut,
        reinterpret_cast<cf*>(theta_dot_out),
        e_mean_out,
        e_var_out,
        ess_out,
        samples_out,
        logq_out,
        log_gauge_out,
        logpsi_out,
        e_loc_out,
        reinterpret_cast<cf*>(o_rows_host),
        session->linalg()
    );
    return err_state();
}

extern "C" qnpeps_e2e_status qnpeps_e2e_step_scratch_bytes(
    const QnpepsE2eConfig* cfg,
    int64_t n_samples,
    const void* terms,
    int64_t host_tile_bytes,
    uint64_t* out_bytes
)
{
    const qnpeps_e2e_status v{check_cfg(cfg)};
    if (v != QNPEPS_E2E_OK) return v;
    if (not out_bytes) return QNPEPS_E2E_ERR_NULL_ARG;
    (void) n_samples;
    (void) terms;
    (void) host_tile_bytes;
    const auto reservation = qnpeps::arena_reservation_bytes();
    if (err_state() != QNPEPS_E2E_OK) return err_state();
    *out_bytes = static_cast<u64>(reservation);
    return QNPEPS_E2E_OK;
}
