#include "core/arena_cursor.cuh"
#include "core/qnpeps_ctx.cuh"
#include "densitymatrix/backend.cuh"
#include "densitymatrix/types.cuh"
#include "dlenv/build.cuh"
#include "linalg/transfer.cuh"
#include "sampler/draw.cuh"
#include "sampler/trunc_svd.cuh"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

namespace qnpeps
{
[[nodiscard]] static auto prepare_density_storage(
    qnpeps_ctx::SamplerState& state, const SamplerConfig& cfg
) -> bool
{
    if (not cfg.density_route) return true;
    auto& density = state.density;
    const auto input_bond = std::max({cfg.chi_s, cfg.chi_c, cfg.dim_bond});
    const auto upper_bond = std::max(cfg.chi_s, cfg.chi_c);
    const auto output_dimension = std::max(cfg.dim_phys * cfg.dim_bond, cfg.dim_bond);
    density.state_value_count = static_cast<usize>(cfg.ly) * static_cast<usize>(input_bond)
                                * static_cast<usize>(cfg.dim_bond) * static_cast<usize>(input_bond);
    density.result_site_stride = static_cast<usize>(upper_bond)
                                 * static_cast<usize>(output_dimension)
                                 * static_cast<usize>(upper_bond);
    density.result_value_count = static_cast<usize>(cfg.ly) * density.result_site_stride;
    density.host_dimensions.resize(3_uz * static_cast<usize>(cfg.ly));
    return true;
}

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
    upload(
        device_ptr,
        reinterpret_cast<const cuFloatComplex*>(host_tensor.data()),
        host_tensor.num_elems()
    );
}

auto upload_to_device(ArenaCursor& arena, const HostTensor& host_tensor) -> cuFloatComplex*
{
    auto* device_ptr = arena.take<cuFloatComplex>(host_tensor.num_elems());
    upload_to_device(device_ptr, host_tensor);
    return device_ptr;
}

static auto carve_sampler_arena(
    qnpeps_ctx::SamplerState& state, const SamplerConfig& cfg, ArenaCursor& arena
) -> void
{
    auto& samp = state.samp;

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

    samp.max_env_above_site() = chi_aux_bond * chi_env_max;
    samp.max_ket_site() = chi_s * dim_phys * dim_bond * chi_s;
    samp.max_env_unsampled() = chi_s * chi_dl * chi_s;
    samp.max_reduce_input() = max_reduced_m * max_reduced_n;
    samp.max_rfactor() = chi_env_max * max_reduced_n;
    samp.max_sketch() = max_reduced_m * chi_env_max;
    samp.max_tmp() = std::max(max_tmp_env, samp.max_reduce_input());
    samp.max_sigma() = chi_s * chi_dl * chi_s;
    samp.max_sigma_full() = dim_phys * dim_phys * samp.max_sigma();
    samp.max_rho() = dim_phys * dim_phys;

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
    {
        samp.ket_row0()[static_cast<usize>(col)] =
            arena.take<cuFloatComplex>(samp.peps_shapes()[0][static_cast<usize>(col)].num_elems());
    }
    state.allocation.unit = arena.take<cuFloatComplex>(1);

    const auto take_array = [&](i64 stride)
    {
        CuSpanCF32 buffer{};
        buffer.stride = stride;
        buffer.p = arena.take<cuFloatComplex>(static_cast<usize>(stride) * dim_batch);
        return buffer;
    };
    samp.env_above()[0] = take_array(samp.max_env_above_site() * cfg.ly);
    samp.env_above()[1] = take_array(samp.max_env_above_site() * cfg.ly);
    samp.ket() = take_array(samp.max_ket_site() * cfg.ly);
    samp.env_unsampled() = take_array(samp.max_env_unsampled() * (cfg.ly + 1));
    samp.sigma() = take_array(samp.max_sigma());
    samp.sigma_full() = take_array(samp.max_sigma_full());
    samp.sigma_full_scratch() = take_array(samp.max_sigma_full());
    samp.rho() = take_array(samp.max_rho());
    samp.rfactor() = take_array(samp.max_rfactor());
    samp.tmp_a() = take_array(samp.max_tmp());
    samp.tmp_b() = take_array(samp.max_tmp());
    samp.reduce_input() = take_array(samp.max_reduce_input());
    samp.sketch() = take_array(samp.max_sketch());
    samp.projection() = take_array(samp.max_rfactor());
    samp.rfactor_next() = take_array(samp.max_rfactor());
    samp.gram() = take_array(chi_env_max * chi_env_max);

    samp.gram_ptrs() = arena.take<cuFloatComplex*>(dim_batch);
    samp.sketch_ptrs() = arena.take<cuFloatComplex*>(dim_batch);
    samp.info() = arena.take<int>(dim_batch);
    samp.fail() = arena.take<int>(1);

    const auto ptr_capacity = dim_batch;
    const auto ptr_slot_count = pointer_slots(num_rows, num_cols);

    state.allocation.ptr_region = arena.take<cuFloatComplex*>(ptr_slot_count * ptr_capacity);
    samp.drawn_spin() = arena.take<int>(dim_batch);
    samp.row_spins() = arena.take<int>(dim_batch * num_cols);
    samp.logpc() = arena.take<f64>(dim_batch);
    samp.lognorm() = arena.take<f64>(dim_batch);
    samp.samples() = arena.take<u8>(dim_batch * num_rows * num_cols);
    state.allocation.device_seed = arena.take<u64>(1);

    if (cfg.density_route)
    {
        state.density.workspace = densitymatrix::take_workspace(
            state.samp.linalg(),
            arena,
            {.num_sites = cfg.ly,
             .input_bond = std::max({cfg.chi_s, cfg.chi_c, cfg.dim_bond}),
             .operator_bond = cfg.dim_bond,
             .output_dimension = std::max(cfg.dim_phys * cfg.dim_bond, cfg.dim_bond),
             .upper_bond = std::max(cfg.chi_s, cfg.chi_c),
             .lanes = 1}
        );
        state.density.state_values = arena.take<cuDoubleComplex>(state.density.state_value_count);
        state.density.result_values = arena.take<cuDoubleComplex>(state.density.result_value_count);
        state.density.result_dimensions = arena.take<i32>(3_uz * num_cols);
        state.density.normalization_log = arena.take<f64>(1);
        state.density.output_gauge = arena.take<f64>(1);
    }
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
        .row_spin_stride = ctx.sampler.execution.dim_batch,
        .fast_mode = ctx.cfg.sampling_mode == QNPEPS_SAMPLING_FAST,
        .chi_c = ctx.cfg.chi_c,
        .seed = ctx.cfg.seed,
        .batch_base = 0,
        .density_route = ctx.cfg.sampler_truncation_route == QNPEPS_TRUNCATION_DENSITY,
        .state_density_cutoff = ctx.cfg.sampler_density_cutoff,
        .projected_density_cutoff = ctx.cfg.projected_density_cutoff,
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
    capacity_cfg.row_spin_stride = allocation.dim_batch_capacity;

    if (not prepare_density_storage(ctx.sampler, capacity_cfg)) return false;

    ArenaCursor* active_cursor{};
    if (scratch)
    {
        allocation.external_cursor = ArenaCursor::carve(scratch, scratch_bytes);
        active_cursor = &allocation.external_cursor;
    }
    else
        active_cursor = &ctx.linalg().persistent_arena();

    allocation.active_cursor = active_cursor;
    ctx.sampler.samp.bind_linalg(ctx.linalg());
    carve_sampler_arena(ctx.sampler, capacity_cfg, *active_cursor);
    if (err_state() != QNPEPS_OK) return false;

    ctx.sampler.samp.bind_arena(*active_cursor);
    ctx.linalg().svd_workspace().bind_arena(*active_cursor);
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
    HostTensor unit{Shape{1, 1, 1, 1}};
    unit.values()[0] = cf32{1.0f, 0.0f};
    upload_to_device(state.allocation.unit, unit);
    return err_state() == QNPEPS_OK;
}

[[nodiscard]] static auto upload_rangefinder_pointer_arrays(Sampler& samp, usize lane_capacity)
    -> bool
{
    std::vector<cuFloatComplex*> host_pointers{};
    host_pointers.resize(lane_capacity);

    const auto upload_pointers = [&](cuFloatComplex** device_pointers,
                                     const CuSpanCF32& array) -> bool
    {
        for (auto lane = 0_uz; lane < lane_capacity; ++lane)
            host_pointers[lane] = array.p + static_cast<i64>(lane) * array.stride;
        upload(device_pointers, host_pointers.data(), lane_capacity);
        return err_state() == QNPEPS_OK;
    };

    if (not upload_pointers(samp.gram_ptrs(), samp.gram())) return false;
    return upload_pointers(samp.sketch_ptrs(), samp.sketch());
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
        {
            const auto idx = slot * lane_capacity + lane;
            host_pointers[idx] = slot_base + static_cast<i64>(lane) * stride;
        }
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
    {
        for (auto col = 0_uz; col < num_cols; ++col)
        {
            fill_broadcast_pointers(
                place_pointer_array(samp.mpo_ptrs()[row][col]), samp.mpo()[row][col]
            );
        }
    }

    samp.ket_row0_ptrs().resize(num_cols);
    for (auto col = 0_uz; col < num_cols; ++col)
    {
        fill_broadcast_pointers(
            place_pointer_array(samp.ket_row0_ptrs()[col]), samp.ket_row0()[col]
        );
    }
    fill_broadcast_pointers(place_pointer_array(samp.dl_unit_ptrs()), state.allocation.unit);

    upload(device_pointer_region, host_pointers.data(), fixed_slots * lane_capacity);
    if (err_state() != QNPEPS_OK) return false;

    samp.dlenv_env_ptrs().assign(num_env_rows, std::vector<cuFloatComplex**>(num_cols, nullptr));
    for (auto row = 0_uz; row < num_env_rows; ++row)
    {
        for (auto col = 0_uz; col < num_cols; ++col)
            place_pointer_array(samp.dlenv_env_ptrs()[row][col]);
    }

    samp.dlenv_sigma_ptrs().assign(num_env_rows, std::vector<cuFloatComplex**>(num_cols, nullptr));
    for (auto row = 0_uz; row < num_env_rows; ++row)
    {
        for (auto col = 0_uz; col < num_cols; ++col)
            place_pointer_array(samp.dlenv_sigma_ptrs()[row][col]);
    }

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

    const auto sampler_truncation = trunc_svd::require_sampler_route();
    if (sampler_truncation == trunc_svd::Route::invalid) return;
    const auto dlenv_truncation = trunc_svd::require_dlenv_route();
    if (dlenv_truncation == trunc_svd::Route::invalid) return;

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
    const auto graph_capturable = not cfg.density_route
                                  and sampler_truncation == trunc_svd::Route::rangefinder
                                  and dlenv_truncation == trunc_svd::Route::rangefinder;
    ctx.use_graph = graph_capturable;

    ctx.sampler.allocation.allocated = true;
    if (not ctx.sampler.ready())
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
        if (lane.sampling_owned) maybe_free_device(lane.sampling);
        lane.sampling = nullptr;
        lane.sampling_owned = false;
    }
    ctx.sampler.samp.permutation_cache().release();
    maybe_free_host(ctx.sampler.staging.h_samples);
    maybe_free_host(ctx.sampler.staging.h_logpc);
    maybe_free_host(ctx.sampler.staging.h_lognorm);
    if (ctx.sampler.execution.graph)
        CUDA_NOCHECK(cudaGraphExecDestroy(ctx.sampler.execution.graph));

    ctx.sampler.allocation = {};
    ctx.sampler.staging = {};
    ctx.sampler.execution = {};
    ctx.sampler.density = {};
}

}
}
