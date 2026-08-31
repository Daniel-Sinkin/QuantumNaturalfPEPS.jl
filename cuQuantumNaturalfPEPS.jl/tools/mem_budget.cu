#include "qnpeps.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
namespace
{
inline constexpr uint64_t k_sample_count{3000};
inline constexpr uint64_t k_sample_batch_size{2048};

[[nodiscard]] auto MiB(size_t b) noexcept -> double
{
    return (double) b / 1048576.0;
}
}
int main(int argc, char** argv)
{
    int L{atoi(argv[1])}, D{atoi(argv[2])};
    QnpepsConfig cfg{};
    cfg.struct_size = sizeof(QnpepsConfig);
    cfg.lx = L;
    cfg.ly = L;
    cfg.dim_phys = 2;
    cfg.dim_bond = D;
    cfg.chi_s = D;
    cfg.chi_dl = D;
    cfg.seed = 1;
    size_t f0, f1, f2a, f2, f3a, f3, tot;
    cudaFree(0);
    cudaMemGetInfo(&f0, &tot);
    long pbytes{qnpeps_peps_bytes(&cfg)};
    void* d_peps{};
    if (cudaMalloc(&d_peps, pbytes))
    {
        printf("%d,%d,OOM_peps\n", L, D);
        return 0;
    }
    std::vector<float> hp{};
    hp.assign((size_t) (pbytes / 4), 0.01f);
    cudaMemcpy(d_peps, hp.data(), pbytes, cudaMemcpyHostToDevice);
    cudaMemGetInfo(&f1, &tot);
    long dlbytes{qnpeps_dlenv_bytes(&cfg)};
    void* d_dl{};
    if (cudaMalloc(&d_dl, dlbytes))
    {
        printf("%d,%d,OOM_dlenv\n", L, D);
        return 0;
    }
    cudaMemGetInfo(&f2a, &tot);
    qnpeps_build_dlenv(&cfg, (const qnpeps_device_peps*) d_peps, (qnpeps_device_dlenv*) d_dl, 0, 0);
    cudaDeviceSynchronize();
    cudaMemGetInfo(&f2, &tot);
    long bb{qnpeps_sample_bytes(&cfg, k_sample_count)};
    void *d_bits{}, *d_lp{};
    cudaMalloc(&d_bits, bb);
    cudaMalloc(&d_lp, (size_t) k_sample_count * sizeof(double));
    int64_t scratch_bytes{qnpeps_sample_scratch_bytes(&cfg, k_sample_batch_size)};
    void* d_scratch{};
    cudaMalloc(&d_scratch, (size_t) scratch_bytes);
    cudaMemGetInfo(&f3a, &tot);
    const QnpepsSampleArgs sample_args{
        .struct_size = sizeof(QnpepsSampleArgs),
        .peps = (const qnpeps_device_peps*) d_peps,
        .dlenv = (const qnpeps_device_dlenv*) d_dl,
        .gpus = 1,
        .scratch = d_scratch,
        .scratch_bytes = (uint64_t) scratch_bytes,
        .samples_out = (uint8_t*) d_bits,
        .log_prob_config = (double*) d_lp,
        .log_gauge = nullptr,
        .n_samples = k_sample_count,
        .batch_base = 0,
        .dim_batch = k_sample_batch_size,
        .stream = nullptr,
    };
    qnpeps_sample(&cfg, &sample_args);
    cudaDeviceSynchronize();
    cudaMemGetInfo(&f3, &tot);
    printf(
        "%d,%d,%.0f,%.0f,%.1f,%.1f,%.1f,%.1f\n",
        L,
        D,
        MiB(tot),
        MiB(tot - f0),
        MiB(f0 - f1),
        MiB(f1 - f2a),
        MiB(f2a - f2),
        MiB(f3a - f3)
    );
    return 0;
}
