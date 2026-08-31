#ifndef QNPEPS_E2E_MG_COMMON_CUH
#define QNPEPS_E2E_MG_COMMON_CUH

#include "capi/qnpeps.h"
#include "common.cuh"
#include "dans_qnpeps_e2e.h"
#include "dans_qnpeps_eloc.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <mutex>
#include <sched.h>
#include <string>
#include <vector>

namespace qn_e2e
{
namespace mg
{
inline constexpr int k_max_gpu_count{64};
inline constexpr int k_peer_cache_size{k_max_gpu_count + 1};
inline constexpr int k_probe_fill{0xAB};
inline constexpr int k_probe_iterations{8};
inline constexpr usize k_probe_bytes{static_cast<usize>(1) << 20};

[[nodiscard]] inline auto sampler_config(const QnpepsE2eConfig& config) noexcept -> QnpepsConfig
{
    QnpepsConfig output{};
    output.struct_size = sizeof(QnpepsConfig);
    output.lx = config.lx;
    output.ly = config.ly;
    output.dim_phys = config.dim_phys;
    output.dim_bond = config.dim_bond;
    output.chi_s = config.chi_s;
    output.chi_dl = config.chi_dl;
    output.seed = config.seed;
    output.sampling_mode = config.sampling_mode;
    output.chi_c = config.contract_dim;
    return output;
}

[[nodiscard]] inline auto local_energy_config(const QnpepsE2eConfig& config) noexcept
    -> QnpepsElocConfig
{
    QnpepsElocConfig output{};
    output.struct_size = sizeof(QnpepsElocConfig);
    output.lx = config.lx;
    output.ly = config.ly;
    output.dim_phys = config.dim_phys;
    output.dim_bond = config.dim_bond;
    output.chi_eo = config.chi_eo;
    output.meo = config.meo;
    return output;
}

inline auto map_sampler_status(qnpeps_status status) noexcept -> void
{
    if (status == QNPEPS_OK) return;
    if (status == QNPEPS_ERR_BAD_CONFIG)
        set_err(QNPEPS_E2E_ERR_BAD_CONFIG);
    else if (status == QNPEPS_ERR_CUDA)
        set_err(QNPEPS_E2E_ERR_CUDA);
    else if (status == QNPEPS_ERR_OOM)
        set_err(QNPEPS_E2E_ERR_OOM);
    else
        set_err(QNPEPS_E2E_ERR_INTERNAL);
}

inline auto map_local_energy_status(qnpeps_eloc_status status) noexcept -> void
{
    if (status == QNPEPS_ELOC_OK) return;
    if (status == QNPEPS_ELOC_ERR_BAD_CONFIG)
        set_err(QNPEPS_E2E_ERR_BAD_CONFIG);
    else if (status == QNPEPS_ELOC_ERR_CUDA)
        set_err(QNPEPS_E2E_ERR_CUDA);
    else if (status == QNPEPS_ELOC_ERR_OOM)
        set_err(QNPEPS_E2E_ERR_OOM);
    else
        set_err(QNPEPS_E2E_ERR_INTERNAL);
}

template <class T>
[[nodiscard]] auto device_allocate(i64 count) noexcept -> T*
{
    if (err_state() != QNPEPS_E2E_OK) return nullptr;
    void* pointer{};
    const auto allocation_count = static_cast<usize>(std::max(count, i64{1}));
    const cudaError_t allocation_status{cudaMalloc(&pointer, sizeof(T) * allocation_count)};
    if (allocation_status != cudaSuccess)
    {
        set_err(QNPEPS_E2E_ERR_OOM);
        return nullptr;
    }
    return static_cast<T*>(pointer);
}

[[nodiscard]] inline auto _config_check(const QnpepsE2eConfig* config) noexcept -> qnpeps_e2e_status
{
    if (not config) return QNPEPS_E2E_ERR_NULL_ARG;
    if (config->struct_size != sizeof(QnpepsE2eConfig)) return QNPEPS_E2E_ERR_BAD_VERSION;
    const auto valid_geometry =
        config->lx >= 2 and config->ly >= 2 and config->dim_phys == 2 and config->dim_bond >= 1;
    if (not valid_geometry) return QNPEPS_E2E_ERR_BAD_CONFIG;
    const auto valid_ranks =
        config->chi_s >= 1 and config->chi_dl >= 1 and config->chi_eo >= 1 and config->meo >= 1;
    if (not valid_ranks) return QNPEPS_E2E_ERR_BAD_CONFIG;
    const auto valid_sampling_mode = config->sampling_mode == QNPEPS_SAMPLING_FAST
                                     or config->sampling_mode == QNPEPS_SAMPLING_FULL;
    if (not valid_sampling_mode) return QNPEPS_E2E_ERR_BAD_CONFIG;
    const auto valid_contract_dimension =
        config->sampling_mode != QNPEPS_SAMPLING_FULL or config->contract_dim >= 1;
    if (not valid_contract_dimension) return QNPEPS_E2E_ERR_BAD_CONFIG;
    const auto valid_sample_batch =
        config->sample_batch >= 0 and config->sample_batch <= qnpeps::k_max_batch_size;
    if (not valid_sample_batch) return QNPEPS_E2E_ERR_BAD_CONFIG;
    return QNPEPS_E2E_OK;
}

[[nodiscard]] inline auto read_integer_file(const std::string& path, int default_value) -> int
{
    std::ifstream input{path};
    int value{};
    if (input >> value) return value;
    return default_value;
}

[[nodiscard]] inline auto gpu_numa_node(int device) -> int
{
    qnpeps::CuArray<char, 32> bus_id{};
    if (cudaDeviceGetPCIBusId(bus_id.data(), sizeof(bus_id), device) != cudaSuccess) return -1;
    std::string normalized_bus_id{bus_id.data()};
    for (auto& character : normalized_bus_id)
    {
        character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    }
    return read_integer_file("/sys/bus/pci/devices/" + normalized_bus_id + "/numa_node", -1);
}

[[nodiscard]] inline auto pin_cpu_list(const std::string& list) noexcept -> int
{
    cpu_set_t cpu_set{};
    CPU_ZERO(&cpu_set);
    bool found_cpu{};
    const auto is_digit = [](char character) noexcept
    { return std::isdigit(static_cast<unsigned char>(character)); };
    usize index{0};
    while (index < list.size())
    {
        if (not is_digit(list[index]))
        {
            ++index;
            continue;
        }
        int range_begin{};
        while (index < list.size() and is_digit(list[index]))
        {
            range_begin = range_begin * 10 + (list[index++] - '0');
        }
        int range_end{range_begin};
        if (index < list.size() and list[index] == '-')
        {
            ++index;
            range_end = 0;
            while (index < list.size() and is_digit(list[index]))
            {
                range_end = range_end * 10 + (list[index++] - '0');
            }
        }
        for (int cpu{range_begin}; cpu <= range_end; ++cpu)
        {
            CPU_SET(cpu, &cpu_set);
            found_cpu = true;
        }
    }
    if (not found_cpu) return -1;
    return sched_setaffinity(0, sizeof(cpu_set), &cpu_set);
}

inline auto pin_thread_to_gpu_numa(int device) -> int
{
    const int numa_node{gpu_numa_node(device)};
    if (numa_node < 0) return -1;
    std::ifstream input{"/sys/devices/system/node/node" + std::to_string(numa_node) + "/cpulist"};
    std::string list;
    std::getline(input, list);
    if (list.empty()) return -1;
    return pin_cpu_list(list);
}

inline auto enable_all_peer_access(int gpu_count) noexcept -> void
{
    for (int source_device{0}; source_device < gpu_count; ++source_device)
    {
        if (cudaSetDevice(source_device) != cudaSuccess) continue;
        for (int destination_device{0}; destination_device < gpu_count; ++destination_device)
        {
            if (source_device == destination_device) continue;
            int can_access{};
            cudaDeviceCanAccessPeer(&can_access, source_device, destination_device);
            if (not can_access) continue;
            const cudaError_t status{cudaDeviceEnablePeerAccess(destination_device, 0)};
            if (status != cudaSuccess and status != cudaErrorPeerAccessAlreadyEnabled)
                cudaGetLastError();
        }
    }
}

[[nodiscard]] inline auto peer_roundtrip(
    int source_device, int destination_device, usize bytes, int iterations
) noexcept -> int
{
    void* source_buffer{};
    void* destination_buffer{};
    if (cudaSetDevice(source_device) != cudaSuccess) return 1;
    if (cudaMalloc(&source_buffer, bytes) != cudaSuccess) return 1;
    const auto destination_status = cudaSetDevice(destination_device);
    const auto destination_allocation_status = destination_status == cudaSuccess
                                                   ? cudaMalloc(&destination_buffer, bytes)
                                                   : cudaErrorInvalidDevice;
    if (destination_status != cudaSuccess or destination_allocation_status != cudaSuccess)
    {
        cudaSetDevice(source_device);
        cudaFree(source_buffer);
        return 1;
    }
    auto* host = static_cast<u8*>(std::malloc(bytes));
    cudaSetDevice(source_device);
    cudaMemset(source_buffer, k_probe_fill, bytes);
    cudaDeviceSynchronize();

    cudaError_t status{cudaSuccess};
    for (int iteration{}; iteration < iterations; ++iteration)
    {
        const cudaError_t copy_status{cudaMemcpyPeer(
            destination_buffer, destination_device, source_buffer, source_device, bytes
        )};
        if (copy_status != cudaSuccess)
        {
            status = copy_status;
            break;
        }
    }
    cudaSetDevice(destination_device);
    const cudaError_t synchronization_status{cudaDeviceSynchronize()};
    if (status == cudaSuccess) status = synchronization_status;
    const cudaError_t download_status{
        host ? cudaMemcpy(host, destination_buffer, bytes, cudaMemcpyDeviceToHost)
             : cudaErrorInvalidValue
    };
    if (status == cudaSuccess) status = download_status;
    int succeeded{status == cudaSuccess ? 1 : 0};
    if (succeeded and host)
    {
        for (auto index = 0_uz; index < bytes; ++index)
        {
            if (host[index] != k_probe_fill)
            {
                succeeded = 0;
                break;
            }
        }
    }
    else
        succeeded = 0;
    std::free(host);
    cudaSetDevice(source_device);
    cudaFree(source_buffer);
    cudaSetDevice(destination_device);
    cudaFree(destination_buffer);
    if (succeeded) return 0;
    return status != cudaSuccess ? 1 : 2;
}

[[nodiscard]] inline auto probe_peer_access_cached(int gpu_count) -> bool
{
    static std::mutex mutex;
    static qnpeps::CuArray<int, k_peer_cache_size> cache;
    static bool initialized{};
    std::lock_guard<std::mutex> lock{mutex};
    if (not initialized)
    {
        for (int& entry : cache)
            entry = -1;
        initialized = true;
    }
    if (gpu_count < 1 or gpu_count > k_max_gpu_count) return false;
    if (cache[static_cast<usize>(gpu_count)] < 0)
    {
        enable_all_peer_access(gpu_count);
        bool succeeded{true};
        for (int gpu{1}; gpu < gpu_count; ++gpu)
        {
            const auto forward_failed =
                peer_roundtrip(0, gpu, k_probe_bytes, k_probe_iterations) != 0;
            const auto reverse_failed =
                peer_roundtrip(gpu, 0, k_probe_bytes, k_probe_iterations) != 0;
            if (forward_failed or reverse_failed) succeeded = false;
        }
        if (not succeeded)
        {
            for (int source_device{0}; source_device < gpu_count; ++source_device)
            {
                cudaSetDevice(source_device);
                for (int destination_device{}; destination_device < gpu_count; ++destination_device)
                {
                    if (source_device != destination_device)
                    {
                        cudaDeviceDisablePeerAccess(destination_device);
                        cudaGetLastError();
                    }
                }
            }
        }
        for (int device{}; device < gpu_count; ++device)
        {
            cudaSetDevice(device);
            cudaGetLastError();
        }
        cudaSetDevice(0);
        cache[static_cast<usize>(gpu_count)] = succeeded ? 1 : 0;
    }
    return cache[static_cast<usize>(gpu_count)] == 1;
}

struct Shard
{
    i64 base{};
    i64 count{};
};

[[nodiscard]] inline auto build_shards(i64 sample_count, int gpu_count, int samples_per_wave)
    -> std::vector<Shard>
{
    std::vector<Shard> shards{static_cast<usize>(gpu_count)};
    const i64 total_waves{(sample_count + samples_per_wave - 1) / samples_per_wave};
    i64 wave_base{};
    for (int gpu{}; gpu < gpu_count; ++gpu)
    {
        const i64 remaining{static_cast<i64>(gpu_count) - gpu};
        const i64 waves_here{(total_waves - wave_base + remaining - 1) / remaining};
        const i64 wave_end{wave_base + waves_here};
        i64 base{wave_base * samples_per_wave};
        i64 end{wave_end * samples_per_wave};
        if (base > sample_count) base = sample_count;
        if (end > sample_count) end = sample_count;
        shards[static_cast<usize>(gpu)].base = base;
        shards[static_cast<usize>(gpu)].count = end - base;
        wave_base = wave_end;
    }
    return shards;
}

}
}

#endif
