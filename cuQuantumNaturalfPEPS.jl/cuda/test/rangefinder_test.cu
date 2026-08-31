#include "capi/qnpeps.h"

#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <vector>

using cf = std::complex<float>;

namespace
{
auto run_case(int rows, int cols, int rank, int batch, uint64_t seed) -> bool
{
    auto rng = std::mt19937_64(seed);
    auto gauss = std::normal_distribution<float>(0.0f, 1.0f);

    const auto rows_z = static_cast<size_t>(rows);
    const auto rank_z = static_cast<size_t>(rank);
    const size_t reduce_input_n{rows_z * cols * batch};
    const size_t q_n{rows_z * rank * batch};
    const size_t r_n{rank_z * cols * batch};

    std::vector<cf> reduce_input{};
    reduce_input.resize(reduce_input_n);
    std::vector<cf> u{};
    u.resize(rows_z * rank);
    std::vector<cf> v{};
    v.resize(rank_z * cols);
    for (auto lane = 0; lane < batch; ++lane)
    {
        for (auto& value : u)
            value = cf{gauss(rng), gauss(rng)};
        for (auto& value : v)
            value = cf{gauss(rng), gauss(rng)};
        auto* il = reduce_input.data() + static_cast<size_t>(lane) * rows * cols;
        for (auto j = 0; j < cols; ++j)
        {
            for (auto i = 0; i < rows; ++i)
            {
                cf acc{};
                for (auto l = 0; l < rank; ++l)
                    acc += u[i + l * rows] * v[l + j * rank];
                il[i + j * rows] = acc;
            }
        }
    }

    void* d_reduce_input{};
    void* d_q{};
    void* d_r{};
    cudaMalloc(&d_reduce_input, reduce_input_n * sizeof(cf));
    cudaMalloc(&d_q, q_n * sizeof(cf));
    cudaMalloc(&d_r, r_n * sizeof(cf));
    cudaMemcpy(
        d_reduce_input, reduce_input.data(), reduce_input_n * sizeof(cf), cudaMemcpyHostToDevice
    );

    const auto scratch_bytes = qnpeps_batched_rangefinder_scratch_bytes(rows, cols, rank, batch);
    void* d_scratch{};
    cudaMalloc(&d_scratch, static_cast<size_t>(scratch_bytes));

    const auto status = qnpeps_batched_rangefinder(
        d_reduce_input,
        rows,
        cols,
        rank,
        batch,
        static_cast<int64_t>(rows) * cols,
        seed,
        d_q,
        static_cast<int64_t>(rows) * rank,
        d_r,
        static_cast<int64_t>(rank) * cols,
        d_scratch,
        static_cast<uint64_t>(scratch_bytes),
        nullptr
    );

    std::vector<cf> q{};
    q.resize(q_n);
    std::vector<cf> r{};
    r.resize(r_n);
    cudaMemcpy(q.data(), d_q, q_n * sizeof(cf), cudaMemcpyDeviceToHost);
    cudaMemcpy(r.data(), d_r, r_n * sizeof(cf), cudaMemcpyDeviceToHost);
    cudaFree(d_reduce_input);
    cudaFree(d_q);
    cudaFree(d_r);
    cudaFree(d_scratch);

    if (status != QNPEPS_OK)
    {
        std::printf(
            "[rf] rows=%d cols=%d rank=%d batch=%d  status=%d (%s)  -> FAIL\n",
            rows,
            cols,
            rank,
            batch,
            status,
            qnpeps_strerror(status)
        );
        return false;
    }

    float max_reconstruct{};
    float max_orthonormal{};
    float reduce_input_scale{};
    for (auto lane = 0; lane < batch; ++lane)
    {
        const auto lane_z = static_cast<size_t>(lane);
        const auto* il = reduce_input.data() + lane_z * rows * cols;
        const auto* ql = q.data() + lane_z * rows * rank;
        const auto* rl = r.data() + lane_z * rank * cols;
        for (auto j = 0; j < cols; ++j)
        {
            for (auto i = 0; i < rows; ++i)
            {
                cf acc{};
                for (auto l = 0; l < rank; ++l)
                    acc += ql[i + l * rows] * rl[l + j * rank];
                const float mag{std::abs(il[i + j * rows])};
                if (mag > reduce_input_scale) reduce_input_scale = mag;
                const float dev{std::abs(acc - il[i + j * rows])};
                if (dev > max_reconstruct) max_reconstruct = dev;
            }
        }
        for (auto a = 0; a < rank; ++a)
        {
            for (auto b = 0; b < rank; ++b)
            {
                cf acc{};
                for (auto i = 0; i < rows; ++i)
                    acc += std::conj(ql[i + a * rows]) * ql[i + b * rows];
                const float target{(a == b) ? 1.0f : 0.0f};
                const float dev{std::abs(acc - cf{target, 0.0f})};
                if (dev > max_orthonormal) max_orthonormal = dev;
            }
        }
    }

    const float rel_reconstruct{
        max_reconstruct / (reduce_input_scale > 0.0f ? reduce_input_scale : 1.0f)
    };
    const bool ok{rel_reconstruct < 1e-3f and max_orthonormal < 1e-3f};
    std::printf(
        "[rf] rows=%d cols=%d rank=%d batch=%d  rel_reconstruct=%.2e  max|Q'Q-I|=%.2e  -> %s\n",
        rows,
        cols,
        rank,
        batch,
        rel_reconstruct,
        max_orthonormal,
        ok ? "PASS" : "FAIL"
    );
    return ok;
}
}

int main()
{
    bool all_ok{true};
    all_ok &= run_case(16, 16, 4, 4, 1);
    all_ok &= run_case(16, 16, 8, 8, 2);
    all_ok &= run_case(20, 16, 8, 8, 3);
    all_ok &= run_case(24, 12, 6, 16, 4);
    all_ok &= run_case(64, 48, 16, 32, 5);
    std::printf(all_ok ? "[rangefinder_test] PASS\n" : "[rangefinder_test] FAIL\n");
    return all_ok ? 0 : 1;
}
