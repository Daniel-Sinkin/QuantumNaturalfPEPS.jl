#include "dlenv/build.cuh"
#include "qnpeps_ctx.cuh"
#include "sampler/draw.cuh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace qnpeps
{
[[nodiscard]] static auto fixed_pointer_slots(usize num_rows, usize num_cols) noexcept -> usize
{
    const auto num_env_rows = num_rows - 1;
    return 2_uz + (num_cols + 1) + num_env_rows * num_cols + num_cols + 1;
}

[[nodiscard]] static auto pointer_slots(usize num_rows, usize num_cols) noexcept -> usize
{
    const auto num_env_rows = num_rows - 1;
    const auto layout_slots = dlenv::k_sampling_layout_count * num_env_rows * num_cols;
    return fixed_pointer_slots(num_rows, num_cols) + layout_slots;
}

static auto upload_to_device(cuFloatComplex* device_ptr, const HostTensor& host_tensor) -> void
{
    const auto bytes = host_tensor.num_elems() * sizeof(cuFloatComplex);
    CUDA_CHECK(cudaMemcpy(device_ptr, host_tensor.data(), bytes, cudaMemcpyHostToDevice));
}

auto upload_to_device(ArenaCursor& arena, const HostTensor& host_tensor) -> cuFloatComplex*
{
    auto* device_ptr = arena.take<cuFloatComplex>(host_tensor.num_elems());
    upload_to_device(device_ptr, host_tensor);
    return device_ptr;
}

[[nodiscard]] static auto carve_sampler_arena(
    qnpeps_ctx::SamplerState& state, const SamplerConfig& cfg, ArenaCursor arena
) -> ArenaCursor
{
    Sampler& samp = state.samp;

    const auto dim_bond = static_cast<i64>(cfg.dim_bond);
    const auto dim_phys = static_cast<i64>(cfg.dim_phys);
    const auto chi_s = static_cast<i64>(cfg.chi_s);
    const auto chi_dl = static_cast<i64>(cfg.chi_dl);
    const auto chi_c = static_cast<i64>(cfg.chi_c);
    const auto chi_env_max = [&]
    {
        auto out = std::max(chi_s, dim_bond);
        if (not cfg.fast_mode) return std::max(out, chi_c);
        return out;
    }();
    const auto chi_aux_bond = chi_env_max * dim_bond;
    const auto max_reduced_n = std::max(chi_aux_bond, chi_dl * dim_bond * dim_bond);
    const auto max_reduced_m = std::max(chi_s * dim_phys * dim_bond, chi_aux_bond);
    const auto max_tmp_env = chi_s * chi_dl * chi_s * dim_phys * dim_bond;

    // clang-format off
    samp.max_env_above_site() = chi_aux_bond * chi_env_max;
    samp.max_ket_site()       = chi_s * dim_phys * dim_bond * chi_s;
    samp.max_env_unsampled()  = chi_s * chi_dl * chi_s;
    samp.max_reduce_input()   = max_reduced_m * max_reduced_n;
    samp.max_rfactor()        = chi_env_max * max_reduced_n;
    samp.max_sketch()         = max_reduced_m * chi_env_max;
    samp.max_tmp()            = std::max(max_tmp_env, samp.max_reduce_input());
    samp.max_sigma()          = chi_s * chi_dl * chi_s;
    samp.max_sigma_full()     = dim_phys * dim_phys * samp.max_sigma();
    samp.max_rho()            = dim_phys * dim_phys;
    // clang-format on

    const auto dim_batch = static_cast<usize>(cfg.dim_batch);
    const auto num_rows = static_cast<usize>(cfg.lx);
    const auto num_cols = static_cast<usize>(cfg.ly);

    const auto peps_site_shape = [&](int row, int col) -> Shape
    {
        return Shape{
            bond_dim(cfg.ly, col, cfg.dim_bond),
            bond_dim(cfg.lx, row + 1, cfg.dim_bond),
            bond_dim(cfg.ly, col + 1, cfg.dim_bond),
            bond_dim(cfg.lx, row, cfg.dim_bond),
            cfg.dim_phys
        };
    };

    samp.mpo().assign(num_rows, std::vector<cuFloatComplex*>(num_cols, nullptr));
    samp.peps_shapes().assign(num_rows, std::vector<Shape>(num_cols));
    for (auto row = 0; row < cfg.lx; ++row)
    {
        for (auto col = 0; col < cfg.ly; ++col)
        {
            const auto row_u = static_cast<usize>(row);
            const auto col_u = static_cast<usize>(col);
            const auto shape = peps_site_shape(row, col);
            samp.peps_shapes()[row_u][col_u] = shape;
            samp.mpo()[row_u][col_u] = arena.take<cuFloatComplex>(shape.num_elems());
        }
    }
    samp.ket_row0().assign(num_cols, nullptr);
    for (auto col = 0; col < cfg.ly; ++col)
        samp.ket_row0()[static_cast<usize>(col)] =
            arena.take<cuFloatComplex>(samp.peps_shapes()[0][static_cast<usize>(col)].num_elems());
    state.allocation.unit = arena.take<cuFloatComplex>(1);

    const auto take_array = [&](i64 stride)
    {
        CuArray buffer{};
        buffer.stride = stride;
        buffer.p = arena.take<cuFloatComplex>(static_cast<usize>(stride) * dim_batch);
        return buffer;
    };

    // clang-format off
    samp.env_above()[0]       = take_array(samp.max_env_above_site() * cfg.ly);
    samp.env_above()[1]       = take_array(samp.max_env_above_site() * cfg.ly);
    samp.ket()                = take_array(samp.max_ket_site() * cfg.ly);
    samp.env_unsampled()      = take_array(samp.max_env_unsampled() * (cfg.ly + 1));
    samp.sigma()              = take_array(samp.max_sigma());
    samp.sigma_full()         = take_array(samp.max_sigma_full());
    samp.sigma_full_scratch() = take_array(samp.max_sigma_full());
    samp.rho()                = take_array(samp.max_rho());
    samp.rfactor()            = take_array(samp.max_rfactor());
    samp.tmp_a()              = take_array(samp.max_tmp());
    samp.tmp_b()              = take_array(samp.max_tmp());
    samp.reduce_input()       = take_array(samp.max_reduce_input());
    samp.sketch()             = take_array(samp.max_sketch());
    samp.projection()         = take_array(samp.max_rfactor());
    samp.rfactor_next()       = take_array(samp.max_rfactor());
    samp.gram()               = take_array(chi_env_max * chi_env_max);

    samp.gram_ptrs()   = arena.take<cuFloatComplex*>(dim_batch);
    samp.sketch_ptrs() = arena.take<cuFloatComplex*>(dim_batch);
    samp.info()        = arena.take<int>(dim_batch);
    samp.fail()        = arena.take<int>(1);
    // clang-format on

    const auto ptr_capacity = dim_batch;
    const auto ptr_slot_count = pointer_slots(num_rows, num_cols);

    // clang-format off
    state.allocation.ptr_region  = arena.take<cuFloatComplex*>(ptr_slot_count * ptr_capacity);
    samp.drawn_spin()            = arena.take<int>(dim_batch);
    samp.row_spins()             = arena.take<int>(dim_batch * num_cols);
    samp.logpc()                 = arena.take<f64>(dim_batch);
    samp.lognorm()               = arena.take<f64>(dim_batch);
    samp.samples()               = arena.take<u8>(dim_batch * num_rows * num_cols);
    state.allocation.device_seed = arena.take<u64>(1);
    // clang-format on

    return arena;
}

[[nodiscard]] static auto omega_region_bytes(const SamplerConfig& cfg) -> usize
{
    const auto num_rows = static_cast<usize>(cfg.lx);
    const auto num_cols = static_cast<usize>(cfg.ly);
    usize total{};

    std::map<std::pair<usize, usize>, char> seen;
    std::vector<int> bond_above{};
    bond_above.assign(num_cols + 1, 1);
    for (auto col = 0_uz; col <= num_cols; ++col)
        bond_above[col] = bond_dim(cfg.ly, static_cast<int>(col), cfg.dim_bond);

    for (auto row = 1_uz; row < num_rows; ++row)
    {
        std::vector<int> ket_bonds{};
        ket_bonds.assign(num_cols + 1, 1);
        int k{1};
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const int bond_down{bond_dim(cfg.lx, static_cast<int>(row) + 1, cfg.dim_bond)};
            const int bond_right{bond_dim(cfg.ly, static_cast<int>(col) + 1, cfg.dim_bond)};
            const int reduce_rows{k * cfg.dim_phys * bond_down};
            const auto reduce_cols = bond_above[col + 1] * bond_right;
            const int k_next{std::max(1, std::min({cfg.chi_s, reduce_rows, reduce_cols}))};
            ket_bonds[col + 1] = k_next;
            k = k_next;
        }
        ket_bonds[num_cols] = 1;
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const int bond_right{bond_dim(cfg.ly, static_cast<int>(col) + 1, cfg.dim_bond)};
            const auto bond_above_right = static_cast<usize>(bond_above[col + 1]);
            const auto bond_right_u = static_cast<usize>(bond_right);
            const auto omega_rows = bond_above_right * bond_right_u;
            const auto omega_cols = static_cast<usize>(ket_bonds[col + 1]);
            if (seen.emplace(std::make_pair(omega_rows, omega_cols), 0).second)
            {
                total += device_align(omega_rows * omega_cols * sizeof(cuFloatComplex));
            }
        }
        if (cfg.fast_mode or row + 1 == num_rows)
        {
            bond_above = ket_bonds;
            continue;
        }

        auto full_bonds = std::vector<int>(num_cols + 1, 1);
        k = 1;
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto bond_down = bond_dim(cfg.lx, static_cast<int>(row) + 1, cfg.dim_bond);
            const auto bond_right = bond_dim(cfg.ly, static_cast<int>(col) + 1, cfg.dim_bond);

            const auto reduce_rows = k * bond_down;
            const auto reduce_cols = bond_above[col + 1] * bond_right;

            const auto k_next = std::max(1, std::min({cfg.chi_c, reduce_rows, reduce_cols}));
            full_bonds[col + 1] = k_next;
            k = k_next;
            if (seen.emplace(std::make_pair(reduce_cols, k_next), 0).second)
            {
                total += device_align(
                    static_cast<usize>(reduce_cols) * static_cast<usize>(k_next)
                    * sizeof(cuFloatComplex)
                );
            }
        }
        full_bonds[num_cols] = 1;
        bond_above = std::move(full_bonds);
    }
    return total;
}

[[nodiscard]] static auto make_sampler_config(const qnpeps_ctx& ctx) -> SamplerConfig
{
    return {
        .lx = ctx.cfg.lx,
        .ly = ctx.cfg.ly,
        .dim_phys = ctx.cfg.dim_phys,
        .dim_bond = ctx.cfg.dim_bond,
        .chi_dl = std::min(ctx.cfg.chi_dl, ctx.cfg.dim_bond * ctx.cfg.dim_bond),
        .chi_s = ctx.cfg.chi_s,
        .dim_batch = ctx.sampler.execution.dim_batch,
        .fast_mode = ctx.cfg.sampling_mode == QNPEPS_SAMPLING_FAST,
        .chi_c = ctx.cfg.chi_c,
        .seed = ctx.cfg.seed,
        .batch_base = 0,
    };
}

[[nodiscard]] static auto initialize_sampler_arena(
    qnpeps_ctx& ctx, const SamplerConfig& cfg, void* scratch, usize scratch_bytes
) -> bool
{
    auto capacity_cfg = cfg;
    auto& allocation = ctx.sampler.allocation;
    allocation.dim_batch_capacity = std::max(allocation.dim_batch_capacity, cfg.dim_batch);
    capacity_cfg.dim_batch = allocation.dim_batch_capacity;

    const auto measured = carve_sampler_arena(ctx.sampler, capacity_cfg, ArenaCursor::measure());
    const auto omega_bytes = omega_region_bytes(capacity_cfg);
    if (measured.total() > std::numeric_limits<usize>::max() - omega_bytes)
    {
        set_err(QNPEPS_ERR_OOM);
        return false;
    }
    const auto total = measured.total() + omega_bytes;

    char* base{};
    if (scratch)
    {
        if (total > scratch_bytes)
        {
            set_err(QNPEPS_ERR_OOM);
            return false;
        }
        base = static_cast<char*>(scratch);
        allocation.owned = false;
    }
    else
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&base), total));
        if (not base)
        {
            set_err(QNPEPS_ERR_OOM);
            return false;
        }
        allocation.owned = true;
    }

    allocation.base = base;
    allocation.cursor =
        carve_sampler_arena(ctx.sampler, capacity_cfg, ArenaCursor::carve(base, total));
    if (err_state() != QNPEPS_OK) return false;
    if (allocation.cursor.total() != measured.total())
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return false;
    }
    assert(allocation.cursor.total() == measured.total());

    ctx.sampler.samp.bind_linalg(ctx.linalg());
    ctx.sampler.samp.bind_arena(allocation.cursor);
    return true;
}

[[nodiscard]] static auto initialize_dlenv(
    qnpeps_ctx& ctx, const DlEnvView& dlenv_view, usize num_rows, usize num_cols
) -> bool
{
    auto& host_rows = ctx.sampler.samp.dlenv_host();
    const auto num_env_rows = num_rows - 1;
    host_rows.resize(num_env_rows);

    usize dims_offset{};
    for (auto row = 0_uz; row < num_env_rows; ++row)
    {
        auto& site_shapes = host_rows[row].site_shapes;
        site_shapes.resize(num_cols);
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            const auto bond_left = dlenv_view.dims[dims_offset + k_dl_bond_left];
            const auto ket_dim = dlenv_view.dims[dims_offset + k_dl_ket];
            const auto bra_dim = dlenv_view.dims[dims_offset + k_dl_bra];
            const auto bond_right = dlenv_view.dims[dims_offset + k_dl_bond_right];
            dims_offset += k_dl_axis_count;
            site_shapes[col] = Shape{bond_left, ket_dim, bra_dim, bond_right};
        }
    }

    dlenv::ensure_sampling_buffers(ctx);
    if (err_state() != QNPEPS_OK) return false;
    auto& active_lane = ctx.dlenv.lanes[ctx.dlenv.active_lane];
    dlenv::materialize_sampling_buffer(ctx, dlenv_view.values, active_lane.sampling);
    return err_state() == QNPEPS_OK;
}

[[nodiscard]] static auto initialize_unit_tensor(qnpeps_ctx::SamplerState& state) -> bool
{
    auto unit = HostTensor{Shape{1, 1, 1, 1}};
    unit.values()[0] = cf32{1.0f, 0.0f};
    upload_to_device(state.allocation.unit, unit);
    return err_state() == QNPEPS_OK;
}

[[nodiscard]] static auto upload_rangefinder_pointer_arrays(Sampler& samp, usize lane_capacity)
    -> bool
{
    std::vector<cuFloatComplex*> host_pointers{};
    host_pointers.resize(lane_capacity);

    const auto upload = [&](cuFloatComplex** device_pointers, const CuArray& array) -> bool
    {
        for (auto lane = 0_uz; lane < lane_capacity; ++lane)
            host_pointers[lane] = array.p + static_cast<i64>(lane) * array.stride;
        CUDA_CHECK(cudaMemcpy(
            device_pointers,
            host_pointers.data(),
            lane_capacity * sizeof(cuFloatComplex*),
            cudaMemcpyHostToDevice
        ));
        return err_state() == QNPEPS_OK;
    };

    if (not upload(samp.gram_ptrs(), samp.gram())) return false;
    return upload(samp.sketch_ptrs(), samp.sketch());
}

[[nodiscard]] static auto initialize_contraction_pointer_arrays(
    qnpeps_ctx::SamplerState& state, usize num_rows, usize num_cols, usize lane_capacity
) -> bool
{
    auto& samp = state.samp;
    const auto num_env_rows = num_rows - 1;
    const auto fixed_slots = fixed_pointer_slots(num_rows, num_cols);
    auto* device_pointer_region = state.allocation.ptr_region;

    std::vector<cuFloatComplex*> host_pointers{};
    host_pointers.resize(fixed_slots * lane_capacity);
    usize pointer_slot{};
    const auto place_pointer_array = [&](cuFloatComplex**& device_pointers) -> usize
    {
        device_pointers = device_pointer_region + pointer_slot * lane_capacity;
        const auto placed_slot = pointer_slot;
        pointer_slot += 1;
        return placed_slot;
    };
    const auto fill_strided_pointers = [&](usize slot, cuFloatComplex* slot_base, i64 stride)
    {
        for (auto lane = 0_uz; lane < lane_capacity; ++lane)
            host_pointers[slot * lane_capacity + lane] =
                slot_base + static_cast<i64>(lane) * stride;
    };
    const auto fill_broadcast_pointers = [&](usize slot, cuFloatComplex* slot_base)
    {
        for (auto lane = 0_uz; lane < lane_capacity; ++lane)
            host_pointers[slot * lane_capacity + lane] = slot_base;
    };

    fill_strided_pointers(
        place_pointer_array(samp.tmp_a_ptrs()), samp.tmp_a().p, samp.tmp_a().stride
    );
    fill_strided_pointers(
        place_pointer_array(samp.tmp_b_ptrs()), samp.tmp_b().p, samp.tmp_b().stride
    );
    samp.envu_ptrs().resize(num_cols + 1);
    for (auto boundary = 0_uz; boundary <= num_cols; ++boundary)
    {
        const auto site_offset = static_cast<i64>(boundary) * samp.max_env_unsampled();
        fill_strided_pointers(
            place_pointer_array(samp.envu_ptrs()[boundary]),
            samp.env_unsampled().p + site_offset,
            samp.env_unsampled().stride
        );
    }

    samp.mpo_ptrs().assign(num_rows, std::vector<cuFloatComplex**>(num_cols, nullptr));
    for (auto row = 1_uz; row < num_rows; ++row)
        for (auto col = 0_uz; col < num_cols; ++col)
            fill_broadcast_pointers(
                place_pointer_array(samp.mpo_ptrs()[row][col]), samp.mpo()[row][col]
            );

    samp.ket_row0_ptrs().resize(num_cols);
    for (auto col = 0_uz; col < num_cols; ++col)
        fill_broadcast_pointers(
            place_pointer_array(samp.ket_row0_ptrs()[col]), samp.ket_row0()[col]
        );
    fill_broadcast_pointers(place_pointer_array(samp.dl_unit_ptrs()), state.allocation.unit);

    CUDA_CHECK(cudaMemcpy(
        device_pointer_region,
        host_pointers.data(),
        fixed_slots * lane_capacity * sizeof(cuFloatComplex*),
        cudaMemcpyHostToDevice
    ));
    if (err_state() != QNPEPS_OK) return false;

    samp.dlenv_env_ptrs().assign(num_env_rows, std::vector<cuFloatComplex**>(num_cols, nullptr));
    for (auto row = 0_uz; row < num_env_rows; ++row)
        for (auto col = 0_uz; col < num_cols; ++col)
            place_pointer_array(samp.dlenv_env_ptrs()[row][col]);

    samp.dlenv_sigma_ptrs().assign(num_env_rows, std::vector<cuFloatComplex**>(num_cols, nullptr));
    for (auto row = 0_uz; row < num_env_rows; ++row)
        for (auto col = 0_uz; col < num_cols; ++col)
            place_pointer_array(samp.dlenv_sigma_ptrs()[row][col]);

    const auto expected_slots = pointer_slots(num_rows, num_cols);
    if (pointer_slot != expected_slots)
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return false;
    }
    assert(pointer_slot == expected_slots);
    return true;
}

[[nodiscard]] static auto allocate_sampler_staging(
    qnpeps_ctx::SamplerState& state, usize lane_capacity, int num_sites
) -> bool
{
    const auto sample_capacity = lane_capacity * static_cast<usize>(num_sites);
    CUDA_CHECK(
        cudaHostAlloc(&state.staging.h_samples, sample_capacity * sizeof(u8), cudaHostAllocDefault)
    );
    CUDA_CHECK(
        cudaHostAlloc(&state.staging.h_logpc, lane_capacity * sizeof(f64), cudaHostAllocDefault)
    );
    CUDA_CHECK(
        cudaHostAlloc(&state.staging.h_lognorm, lane_capacity * sizeof(f64), cudaHostAllocDefault)
    );
    return err_state() == QNPEPS_OK;
}

namespace sampler
{
auto ctx_sampler_setup(qnpeps_ctx& ctx, const DlEnvView* dlenv, void* scratch, usize scratch_bytes)
    -> void
{
    if (ctx.sampler.allocation.allocated) return;
    if (not dlenv or not dlenv->values)
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return;
    }

    auto& state = ctx.sampler;
    auto& samp = state.samp;
    const auto cfg = make_sampler_config(ctx);
    samp.cfg() = cfg;

    if (not initialize_sampler_arena(ctx, cfg, scratch, scratch_bytes)) return;
    const auto lane_capacity = static_cast<usize>(state.allocation.dim_batch_capacity);
    const auto num_rows = static_cast<usize>(cfg.lx);
    const auto num_cols = static_cast<usize>(cfg.ly);

    if (not initialize_dlenv(ctx, *dlenv, num_rows, num_cols)) return;
    if (not initialize_unit_tensor(state)) return;

    if (not upload_rangefinder_pointer_arrays(samp, lane_capacity)) return;
    if (not initialize_contraction_pointer_arrays(state, num_rows, num_cols, lane_capacity)) return;

    if (not allocate_sampler_staging(state, lane_capacity, cfg.num_sites())) return;
    ctx.use_graph = true;

    ctx.sampler.allocation.allocated = true;
    const auto ready = ctx.sampler.ready();
    if (not ready)
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return;
    }
}

auto ctx_sampler_free(qnpeps_ctx& ctx) -> void
{
    for (auto& lane : ctx.dlenv.lanes)
    {
        if (not lane.sampling) continue;
        CUDA_NOCHECK(cudaFree(lane.sampling));
        lane.sampling = nullptr;
    }
    ctx.sampler.samp.permutation_cache().release();
    if (ctx.sampler.allocation.owned and ctx.sampler.allocation.base)
        CUDA_NOCHECK(cudaFree(ctx.sampler.allocation.base));
    if (ctx.sampler.staging.h_samples) CUDA_NOCHECK(cudaFreeHost(ctx.sampler.staging.h_samples));
    if (ctx.sampler.staging.h_logpc) CUDA_NOCHECK(cudaFreeHost(ctx.sampler.staging.h_logpc));
    if (ctx.sampler.staging.h_lognorm) CUDA_NOCHECK(cudaFreeHost(ctx.sampler.staging.h_lognorm));
    if (ctx.sampler.execution.graph)
        CUDA_NOCHECK(cudaGraphExecDestroy(ctx.sampler.execution.graph));

    ctx.sampler.allocation = {};
    ctx.sampler.staging = {};
    ctx.sampler.execution = {};
}

auto sample_arena_bytes(const QnpepsConfig& config, int max_dim_batch) -> int64_t
{
    const auto cfg = SamplerConfig{
        .lx = config.lx,
        .ly = config.ly,
        .dim_phys = config.dim_phys,
        .dim_bond = config.dim_bond,
        .chi_dl = std::min(config.chi_dl, config.dim_bond * config.dim_bond),
        .chi_s = config.chi_s,
        .dim_batch = std::max(max_dim_batch, 1),
        .fast_mode = config.sampling_mode == QNPEPS_SAMPLING_FAST,
        .chi_c = config.chi_c,
    };
    qnpeps_ctx::SamplerState state{};
    const auto measured = carve_sampler_arena(state, cfg, ArenaCursor::measure());
    const auto omega_bytes = omega_region_bytes(cfg);
    if (measured.total() > std::numeric_limits<usize>::max() - omega_bytes)
    {
        set_err(QNPEPS_ERR_OOM);
        return 0;
    }
    return static_cast<int64_t>(measured.total() + omega_bytes);
}
}
}
