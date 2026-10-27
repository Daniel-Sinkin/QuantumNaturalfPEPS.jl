#pragma once

#include "common.cuh"
#include "core/arena_cursor.cuh"
#include "core/complex.cuh"
#include "core/defer.cuh"
#include "core/session.cuh"
#include "density.cuh"
#include "eloc_kernels.cuh"
#include "../linalg/eo.cuh"
#include "capi/qnpeps.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <new>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace qnpeps;

namespace qn_eloc::env
{

__global__ auto cu_gather(
    cf* out,
    const cf* in,
    const int* gather_indices,
    int n,
    i64 stride_out,
    i64 stride_in,
    int conjugate,
    int dim_batch
) -> void;

auto perm_key(const std::vector<int>& dims, const std::vector<int>& perm) -> std::string;

struct PermutationCache
{
    std::map<std::string, int*> cache;
    auto get(const std::vector<int>& dims, const std::vector<int>& perm) -> int*
    {
        const auto key = perm_key(dims, perm);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
        const auto gather_indices = permutation_index_map(dims, perm);
        if (gather_indices.size() > std::numeric_limits<usize>::max() / sizeof(int))
        {
            qn::set_err(QNPEPS_ELOC_ERR_OOM);
            return nullptr;
        }
        const auto bytes = usize{sizeof(int) * gather_indices.size()};
        int* device_ptr{};
        CUDA_CHECK(cudaMalloc(&device_ptr, bytes));
        CUDA_CHECK(cudaMemcpy(device_ptr, gather_indices.data(), bytes, cudaMemcpyHostToDevice));
        cache.emplace(key, device_ptr);
        return device_ptr;
    }
    auto release() -> void
    {
        for (auto& entry : cache)
            maybe_free_device(entry.second);
        cache.clear();
    }
};

struct PermuteOp
{
    EoDeviceBuffer dst{};
    EoDeviceBuffer src{};
    std::vector<int> dims_in{};
    std::vector<int> perm{};
    int batch{};
    const char* kind{"perm"};
    int m{};
    int n{};
    int k{};
};

auto device_permute(Linalg& linalg, PermutationCache& lookup_tables, const PermuteOp& op) -> void;

__global__ auto cu_project_slab(
    const cf* site,
    const u8* samples,
    i64 lane_stride,
    int site_idx,
    i64 slab,
    cf* out,
    i64 out_stride,
    int lanes
) -> void;

__global__ auto cu_scale_last(cf* site, i64 stride, i64 n, const cf* r, i64 rstride, int lanes)
    -> void;

__global__ auto cu_log_finish(
    const cf* acc,
    i64 astride,
    const f64* f_top,
    const f64* f_down,
    const f64* f_ov,
    f64* out,
    int lanes
) -> void;

struct Shape
{
    int lx{};
    int ly{};
    int dim_phys{};
    int dim_bond{};
    int chi{};
    int lanes{};
    bool density{};
    f64 density_cutoff{};
    RangefinderRoute rangefinder{};
    f64 svd_cutoff{};
};

inline auto shape_for(const QnpepsElocConfig& cfg, int lanes, bool transposed) -> Shape
{
    const auto route = cfg.truncation_route == QNPEPS_TRUNCATION_HOUSEHOLDER
                           ? RangefinderRoute::householder
                       : cfg.truncation_route == QNPEPS_TRUNCATION_SVD ? RangefinderRoute::gesvda
                                                                       : RangefinderRoute::cholqr;
    return Shape{
        transposed ? cfg.ly : cfg.lx,
        transposed ? cfg.lx : cfg.ly,
        cfg.dim_phys,
        cfg.dim_bond,
        cfg.chi_eo,
        lanes,
        cfg.truncation_route == QNPEPS_TRUNCATION_DENSITY,
        cfg.density_cutoff,
        route,
        cfg.svd_cutoff
    };
}

auto arena_product(std::initializer_list<std::uint64_t> factors, usize& result) -> bool;

auto arena_sum(usize left, usize right, usize& result) -> bool;

auto arena_accumulate(usize& total, usize term) -> bool;

auto arena_align(usize value, usize& result) -> bool;

auto arena_slot(std::initializer_list<std::uint64_t> factors, i64& result) -> bool;

auto arena_int(std::initializer_list<std::uint64_t> factors, int& result) -> bool;

auto arena_i64_sum(i64 left, i64 right, i64& result) -> bool;

using Carver = qnpeps::ArenaCursor;

using ContextArena = qnpeps::ArenaCursor;

struct Worker
{
    explicit Worker(Linalg& linalg) : la(linalg) {}

    Shape sh{};
    bool want_h{};
    bool want_fb{};
    Linalg& la;
    PermutationCache permutation_index_maps{};
    ContextArena* arena{};
    Carver scratch{};
    std::map<i64, cf*> omegas{};
    qn_eloc::density::Context* density{};

    const cf* peps{};
    const u8* samples{};
    std::vector<i64> site_off{};

    EoDeviceBuffer eb{};
    qnpeps::CuArray<EoDeviceBuffer, 2> et{};
    EoDeviceBuffer proj{};
    EoDeviceBuffer tmp_a{};
    EoDeviceBuffer tmp_b{};
    EoDeviceBuffer tmp_c{};
    EoDeviceBuffer rroll{};
    EoDeviceBuffer rnext{};
    EoDeviceBuffer sketch{};
    EoDeviceBuffer proj_rf{};
    EoDeviceBuffer gram{};
    cf** gram_ptrs{};
    cf** sketch_ptrs{};
    int* info{};
    int* fail_flag{};
    char* qr_scratch{};
    usize qr_scratch_bytes{};
    f64* f_down{};
    f64* f_top{};
    f64* f_ov{};

    i64 slot_env{};
    i64 slot_proj{};
    i64 slot_tmp{};
    i64 slot_r{};
    i64 slot_rf{};

    std::vector<std::vector<int>> ks_down{};
    std::vector<std::vector<int>> ks_top{};

    EoDeviceBuffer et_full{};
    EoDeviceBuffer hl{};
    EoDeviceBuffer hr{};
    EoDeviceBuffer fbl{};
    EoDeviceBuffer fbr{};
    EoDeviceBuffer proj2{};
    cf* unit{};
    f64* f_hl{};
    f64* f_hr{};
    f64* f_fbl{};
    f64* f_fbr{};
    f64* f_chain{};
    i64 slot_h{};
    i64 slot_fb{};
    std::vector<std::vector<int>> ks_full{};
};

inline auto bd(int axis_len, int pos, int dim_bond) -> int
{
    return bond_dim(axis_len, pos, dim_bond);
}

inline auto omega_host(int n, int k) -> std::vector<cf>
{
    const auto seed = u64{0x777ull ^ (static_cast<u64>(n) << 20) ^ static_cast<u64>(k)};
    auto rng = std::mt19937_64(seed);
    auto gauss = std::normal_distribution<f32>(0.0f, 1.0f);
    usize count{};
    if (n < 1 or k < 1
        or not arena_product({static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(k)}, count))
        return {};
    auto host = std::vector<cf>(count);
    for (auto& z : host)
        z = cf{gauss(rng), gauss(rng)};
    return host;
}

inline auto omega_for(Worker& w, int n, int k) -> cf*
{
    const auto key = i64{static_cast<i64>(n) * 1000000 + k};
    auto it = w.omegas.find(key);
    if (it != w.omegas.end()) return it->second;
    const auto host = std::vector<cf>{omega_host(n, k)};
    if (host.empty()) return nullptr;
    if (not w.arena) return nullptr;
    const auto bytes = host.size() * sizeof(cf);
    auto* d = w.arena->take<cf>(host.size());
    CUDA_CHECK(cudaMemcpy(d, host.data(), bytes, cudaMemcpyHostToDevice));
    w.omegas[key] = d;
    return d;
}

inline auto eb_site(Worker& w, int stack_k, int col) -> cf*
{
    return w.eb.p + (static_cast<i64>(stack_k) * w.sh.ly + col) * w.slot_env * w.sh.lanes;
}

inline auto et_site(Worker& w, int buf, int col) -> cf*
{
    return w.et[static_cast<usize>(buf)].p + static_cast<i64>(col) * w.slot_env * w.sh.lanes;
}

inline auto proj_site(Worker& w, int col) -> cf*
{
    return w.proj.p + static_cast<i64>(col) * w.slot_proj * w.sh.lanes;
}

inline auto etf_site(Worker& w, int k, int col) -> cf*
{
    return w.et_full.p + (static_cast<i64>(k) * w.sh.ly + col) * w.slot_env * w.sh.lanes;
}

auto carve_worker(Worker& w, const Shape& sh, bool want_h, bool want_fb, Carver& cv) -> usize;

auto worker_setup(
    Worker& w,
    const Shape& sh,
    bool want_h,
    bool want_fb,
    const cf* peps,
    const u8* samples,
    cudaStream_t stream,
    ContextArena& context_arena
) -> void;

auto worker_teardown(Worker& w) -> void;

auto worker_begin_rangefinders(Worker& w) -> void;

auto worker_rangefinder_failed(Worker& w) -> int;

auto worker_ensure(
    Worker& w,
    const Shape& sh,
    bool want_h,
    bool want_fb,
    const cf* peps,
    const u8* samples,
    cudaStream_t stream,
    ContextArena& context_arena
) -> void;

auto project_row(Worker& w, int row) -> void;

auto build_env_row(
    Worker& w,
    int row,
    bool contract_up,
    const std::vector<int>* ks_adj,
    const std::function<cf*(int)>& adj_site,
    const std::function<cf*(int)>& out_site,
    f64* f_lane,
    std::vector<int>& ks_out
) -> void;

auto compute_logpsi(Worker& w, f64* logpsi_out, bool full_top) -> void;

struct BT
{
    cf* p{};
    i64 stride{};
    std::vector<int> dim{};
};

inline auto bt_elems(const std::vector<int>& dim) -> i64
{
    auto n = i64{1};
    for (const int d : dim)
    {
        if (d < 0
            or not arena_slot({static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(d)}, n))
            return 0;
    }
    return n;
}

auto bcontract(
    Worker& w,
    Carver& frame,
    const BT& A,
    const std::vector<int>& ca,
    const BT& B,
    const std::vector<int>& cb,
    cf* out_p,
    i64 out_stride,
    int dim_batch = 0
) -> BT;

auto fold_left(
    Worker& w,
    Carver frame,
    const BT& h,
    const BT& top,
    const BT& pr,
    const BT& bot,
    cf* out,
    i64 out_stride,
    int dim_batch = 0
) -> std::vector<int>;

auto fold_right(
    Worker& w,
    Carver frame,
    const BT& h,
    const BT& top,
    const BT& pr,
    const BT& bot,
    cf* out,
    i64 out_stride
) -> std::vector<int>;

auto fold_left4(
    Worker& w,
    Carver frame,
    const BT& h,
    const BT& top,
    const BT& pri,
    const BT& prj,
    const BT& bot,
    cf* out,
    i64 out_stride,
    int dim_batch = 0
) -> std::vector<int>;

auto fold_right4(
    Worker& w,
    Carver frame,
    const BT& h,
    const BT& top,
    const BT& pri,
    const BT& prj,
    const BT& bot,
    cf* out,
    i64 out_stride,
    int dim_batch = 0
) -> std::vector<int>;

auto project_column_into(Worker& w, int row, int col, cf* out, i64 out_stride) -> void;

auto project_row_into(Worker& w, int row, cf* base, i64 slot) -> void;

inline auto hl_site(Worker& w, int row, int k) -> cf*
{
    return w.hl.p + (static_cast<i64>(row) * (w.sh.ly - 1) + k) * w.slot_h * w.sh.lanes;
}

inline auto hr_site(Worker& w, int row, int k) -> cf*
{
    return w.hr.p + (static_cast<i64>(row) * (w.sh.ly - 1) + k) * w.slot_h * w.sh.lanes;
}

inline auto fbl_site(Worker& w, int row, int k) -> cf*
{
    return w.fbl.p + (static_cast<i64>(row) * (w.sh.ly - 1) + k) * w.slot_fb * w.sh.lanes;
}

inline auto fbr_site(Worker& w, int row, int k) -> cf*
{
    return w.fbr.p + (static_cast<i64>(row) * (w.sh.ly - 1) + k) * w.slot_fb * w.sh.lanes;
}

inline auto top_dims(Worker& w, int i, int c) -> std::vector<int>
{
    return {
        w.ks_full[static_cast<usize>(i - 1)][static_cast<usize>(c)],
        bd(w.sh.lx, i, w.sh.dim_bond),
        w.ks_full[static_cast<usize>(i - 1)][static_cast<usize>(c) + 1]
    };
}

inline auto bot_dims_h(Worker& w, int i, int c) -> std::vector<int>
{
    const auto k = int{w.sh.lx - i - 2};
    return {
        w.ks_down[static_cast<usize>(k)][static_cast<usize>(c)],
        bd(w.sh.lx, i + 1, w.sh.dim_bond),
        w.ks_down[static_cast<usize>(k)][static_cast<usize>(c) + 1]
    };
}

inline auto bot_dims_fb(Worker& w, int i, int c) -> std::vector<int>
{
    const auto k = int{w.sh.lx - i - 3};
    return {
        w.ks_down[static_cast<usize>(k)][static_cast<usize>(c)],
        bd(w.sh.lx, i + 2, w.sh.dim_bond),
        w.ks_down[static_cast<usize>(k)][static_cast<usize>(c) + 1]
    };
}

inline auto proj_dims(Worker& w, int i, int c) -> std::vector<int>
{
    return {
        bd(w.sh.ly, c, w.sh.dim_bond),
        bd(w.sh.lx, i + 1, w.sh.dim_bond),
        bd(w.sh.ly, c + 1, w.sh.dim_bond),
        bd(w.sh.lx, i, w.sh.dim_bond)
    };
}

inline auto unit3(Worker& w) -> BT
{
    return {w.unit, 1, {1, 1, 1}};
}

inline auto unit4(Worker& w) -> BT
{
    return {w.unit, 1, {1, 1, 1, 1}};
}

auto rail_normalize(Worker& w, cf* rail, i64 n, i64 slot, f64* f_cut, const f64* f_prev) -> void;

inline auto f_hl_cut(Worker& w, int row, int cut) -> f64*
{
    return w.f_hl + (static_cast<i64>(row) * (w.sh.ly - 1) + cut) * w.sh.lanes;
}

inline auto f_hr_cut(Worker& w, int row, int cut) -> f64*
{
    return w.f_hr + (static_cast<i64>(row) * (w.sh.ly - 1) + cut) * w.sh.lanes;
}

inline auto f_fbl_cut(Worker& w, int row, int cut) -> f64*
{
    return w.f_fbl + (static_cast<i64>(row) * (w.sh.ly - 1) + cut) * w.sh.lanes;
}

inline auto f_fbr_cut(Worker& w, int row, int cut) -> f64*
{
    return w.f_fbr + (static_cast<i64>(row) * (w.sh.ly - 1) + cut) * w.sh.lanes;
}

auto build_h_rails(Worker& w, int i) -> void;

auto build_fb_rails(Worker& w, int i) -> void;

auto emit_stage_c(Worker& w, bool want_h, bool want_fb) -> void;

enum class Bucket
{
    horizontal,
    fourbody,
    longer_horizontal
};

struct FlipInst
{
    Bucket bucket{};
    int active_slot{-1};
    int n_flips{};
    qnpeps::CuArray<int, 4> site{};
    qnpeps::CuArray<int, 4> value{};
    int mask_a{};
    int mask_b{};
    f64 coeff_re{};
    f64 coeff_im{};
};

struct ActiveLists
{
    int slots{};
    int max_lanes{};
    int* masks{};
    int* counts{};
    int* indices{};
    bool owns_device{};
    std::vector<int> host_counts{};
};

auto assign_active_slots(std::vector<FlipInst>& terms, bool enabled) -> int;

__global__ auto cu_build_active_lists(
    const u8* samples,
    i64 lane_stride,
    const int* mask_a,
    const int* mask_b,
    int slots,
    int lanes,
    int max_lanes,
    int* indices,
    int* counts
) -> void;

auto active_lists_setup(
    ActiveLists& lists,
    const std::vector<FlipInst>& terms,
    int slots,
    int max_lanes,
    ContextArena& context_arena
) -> void;

auto active_lists_teardown(ActiveLists& lists) -> void;

auto prepare_active_lists(
    ActiveLists& lists, const u8* samples, i64 lane_stride, int lanes, cudaStream_t stream
) -> void;

__global__ auto cu_project_site_val(
    const cf* site,
    const u8* samples,
    i64 lane_stride,
    int site_idx,
    i64 slab,
    int value_mode,
    int fixed_val,
    int dim_phys,
    cf* out,
    i64 out_stride,
    int lanes
) -> void;

__global__ auto cu_project_site_val_indexed(
    const cf* site,
    const u8* samples,
    i64 lane_stride,
    const int* active_indices,
    int site_idx,
    i64 slab,
    int value_mode,
    int fixed_val,
    int dim_phys,
    cf* out,
    i64 out_stride,
    int active
) -> void;

__global__ auto cu_compact_lanes(
    cf* out,
    i64 out_stride,
    const cf* in,
    i64 in_stride,
    const int* active_indices,
    i64 elems,
    int active
) -> void;

__global__ auto cu_diagonal(
    f64* e_loc,
    const u8* samples,
    i64 lane_stride,
    const int* site_a,
    const int* site_b,
    const f64* coeff,
    int n_diag,
    int lanes
) -> void;

__global__ auto cu_reduce_term(
    f64* e_loc,
    const cf* value,
    const f64* f_top,
    const f64* f_down,
    const f64* f_rail_l,
    const f64* f_rail_r,
    const f64* f_chain,
    const f64* logpsi,
    const u8* samples,
    i64 lane_stride,
    int mask_a,
    int mask_b,
    f64 coeff_re,
    f64 coeff_im,
    int lanes
) -> void;

__global__ auto cu_reduce_term_indexed(
    f64* e_loc,
    const cf* value,
    const f64* f_top,
    const f64* f_down,
    const f64* f_rail_l,
    const f64* f_rail_r,
    const f64* f_chain,
    const f64* logpsi,
    const int* active_indices,
    f64 coeff_re,
    f64 coeff_im,
    int active
) -> void;

inline auto top_bt(Worker& w, int i, int c) -> BT
{
    return (i > 0) ? BT{etf_site(w, i - 1, c), w.slot_env, top_dims(w, i, c)} : unit3(w);
}

inline auto bot_bt_h(Worker& w, int i, int c) -> BT
{
    return (i < w.sh.lx - 1) ? BT{eb_site(w, w.sh.lx - i - 2, c), w.slot_env, bot_dims_h(w, i, c)}
                             : unit3(w);
}

inline auto bot_bt_fb(Worker& w, int i, int c) -> BT
{
    return (i < w.sh.lx - 2) ? BT{eb_site(w, w.sh.lx - i - 3, c), w.slot_env, bot_dims_fb(w, i, c)}
                             : unit3(w);
}

inline auto hl_o_dims(Worker& w, int i, int c) -> std::vector<int>
{
    const auto tb = int{i > 0 ? w.ks_full[static_cast<usize>(i - 1)][static_cast<usize>(c)] : 1};
    const auto bb =
        int{i < w.sh.lx - 1 ? w.ks_down[static_cast<usize>(w.sh.lx - i - 2)][static_cast<usize>(c)]
                            : 1};
    return {tb, bd(w.sh.ly, c, w.sh.dim_bond), bb};
}

inline auto hr_o_dims(Worker& w, int i, int c) -> std::vector<int>
{
    const auto tb =
        int{i > 0 ? w.ks_full[static_cast<usize>(i - 1)][static_cast<usize>(c + 1)] : 1};
    const auto bb =
        int{i < w.sh.lx - 1
                ? w.ks_down[static_cast<usize>(w.sh.lx - i - 2)][static_cast<usize>(c + 1)]
                : 1};
    return {tb, bd(w.sh.ly, c + 1, w.sh.dim_bond), bb};
}

auto reduce_term(
    Worker& w,
    int mask_a,
    int mask_b,
    f64 coeff_re,
    f64 coeff_im,
    const cf* value,
    f64* f_top,
    f64* f_down,
    f64* f_rail_l,
    f64* f_rail_r,
    f64* e_loc,
    const f64* logpsi
) -> void;

auto reduce_term_indexed(
    Worker& w,
    const FlipInst& term,
    const cf* value,
    const f64* f_chain,
    const int* active_indices,
    int active,
    f64* f_top,
    f64* f_down,
    f64* f_rail_l,
    f64* f_rail_r,
    f64* e_loc,
    const f64* logpsi
) -> void;

auto launch_diagonal(
    Worker& w, f64* e_loc, const int* site_a, const int* site_b, const f64* coeff, int n_diag
) -> void;

__global__ auto cu_fourbody(
    int lanes, int mid, const cf* cl, i64 stride_l, const cf* cr, i64 stride_r, cf* out
) -> void;

auto launch_dot(
    Worker& w,
    int mid,
    const cf* cl,
    i64 stride_l,
    const cf* cr,
    i64 stride_r,
    cf* out,
    int dim_batch = 0
) -> void;

auto launch_fourbody(Worker& w, int mid, const cf* cl, const cf* cr, cf* out, int dim_batch = 0)
    -> void;

inline auto fb_left_dims(Worker& w, int upper, int k) -> std::vector<int>
{
    const auto t = BT{top_bt(w, upper, k)};
    const auto b = BT{bot_bt_fb(w, upper, k)};
    const auto pi = std::vector<int>{proj_dims(w, upper, k)};
    const auto pj = std::vector<int>{proj_dims(w, upper + 1, k)};
    return {t.dim[2], pi[2], pj[2], b.dim[2]};
}

inline auto fb_right_dims(Worker& w, int upper, int k) -> std::vector<int>
{
    const auto c = int{k + 1};
    const auto t = BT{top_bt(w, upper, c)};
    const auto b = BT{bot_bt_fb(w, upper, c)};
    const auto pi = std::vector<int>{proj_dims(w, upper, c)};
    const auto pj = std::vector<int>{proj_dims(w, upper + 1, c)};
    return {t.dim[0], pi[0], pj[0], b.dim[0]};
}

auto corner_mode(const FlipInst& t, int r, int c, int ly) -> int;

auto project_corner(Worker& w, int r, int c, int mode, cf* out, i64 out_stride) -> BT;

auto project_corner_indexed(
    Worker& w,
    int r,
    int c,
    int mode,
    const int* active_indices,
    int active,
    cf* out,
    i64 out_stride
) -> BT;

auto compact_tensor(Worker& w, const BT& src, const int* active_indices, int active, cf* out) -> BT;

auto chain_normalize(Worker& w, cf* x, i64 n, i64 stride) -> void;

auto chain_normalize_batch(Worker& w, cf* x, i64 n, i64 stride, f64* f_chain, int dim_batch)
    -> void;

auto eval_horizontal(Worker& w, const FlipInst& t, cf* value_out) -> void;

auto eval_horizontal_compact(
    Worker& w, const FlipInst& t, const int* active_indices, int active, cf* value_out
) -> f64*;

auto eval_fourbody(Worker& w, const FlipInst& t, cf* value_out) -> void;

auto eval_fourbody_compact(
    Worker& w, const FlipInst& t, const int* active_indices, int active, cf* value_out
) -> f64*;

auto eval_flip_inst(
    Worker& w,
    const FlipInst& t,
    cf* value_buf,
    f64* e_loc,
    const f64* logpsi,
    const int* active_indices = nullptr,
    int active = -1,
    int compact_min = 2
) -> void;

__global__ auto cu_gscale(
    cf* g,
    const f64* f_top,
    const f64* f_down,
    const f64* f_rail_l,
    const f64* f_rail_r,
    const f64* logpsi,
    int lanes
) -> void;

__global__ auto cu_write_o_block(
    cf* base,
    i64 compact_count,
    i64 compact_offset,
    const cf* src,
    i64 src_stride,
    int slice,
    const cf* g,
    int lanes
) -> void;

__global__ auto cu_u8_to_i32(int* out, const u8* in, i64 total) -> void;

__global__ auto cu_add_lambda_diag(cf* t, int ntot, f64 lambda) -> void;

auto ok_layout(const Shape& sh, std::vector<i64>& off, std::vector<int>& slice) -> i64;

auto emit_o_rows(
    Worker& w,
    cf* wave_base,
    i64 compact_count,
    const std::vector<i64>& compact_off,
    const std::vector<int>& compact_slice,
    const f64* logpsi,
    cf* gscratch
) -> void;

auto site_offsets(int lx, int ly, int dim_bond, int dim_phys) -> std::vector<i64>;

__global__ auto cu_transpose_samples(const u8* samples, u8* out, int lx, int ly, i64 n) -> void;

auto build_transposed_peps(
    const QnpepsElocConfig& cfg, const cf* peps, cf* peps_t, cudaStream_t stream
) -> void;

auto classify_route(
    const QnpepsElocConfig& cfg, const QnpepsElocFlipTerm& ft, FlipInst& out, int& pass
) -> bool;

struct RunCache
{
    explicit RunCache(Linalg& linalg) : w(linalg), wt(linalg) {}

    QnpepsElocConfig cfg{};
    i64 n_samples{};
    uint32_t flags{};
    cudaStream_t stream{};
    std::vector<QnpepsElocDiagBond> diag_terms{};
    std::vector<QnpepsElocFlipTerm> flip_terms{};
    QnpepsElocTermTable term_table{};
    std::vector<i64> ok_off{};
    std::vector<int> ok_slice{};
    i64 compact_count{};
    std::vector<FlipInst> orig_terms{};
    std::vector<FlipInst> tr_terms{};
    bool active_compact{};
    int active_compact_min{};
    ActiveLists orig_active{};
    ActiveLists tr_active{};
    cf* peps_t{};
    u8* samples_t{};
    int* d_a{};
    int* d_b{};
    f64* d_j{};
    cf* value_buf{};
    f64* logpsi_tr{};
    cf* gscratch{};
    int* d_boff{};
    int* d_bslice{};
    int* spins_i32{};
    bool orig_fb{};
    bool orig_hterms{};
    bool tr_fb{};
    bool orig_h{};
    Worker w;
    Worker wt;
    std::vector<std::pair<i64, int>> wave_ranges{};
    ContextArena* context_arena{};
    bool ready{};
};

template <typename T>
auto cache_alloc(RunCache& state, T*& pointer, usize count) -> void
{
    pointer = state.context_arena->take<T>(count);
}

auto run_cache_teardown(RunCache& state) -> void;

auto run_cache_setup(
    RunCache& state,
    const QnpepsElocConfig& cfg,
    i64 n_samples,
    const QnpepsElocTermTable& tt,
    uint32_t flags,
    ContextArena& context_arena
) -> void;

}

inline auto report_rangefinder_failure(int fail) -> void
{
    if (fail > 0)
        qn::set_backend_err(QNPEPS_ELOC_ERR_CUDA, "rangefinder", fail, __FILE__, __LINE__);
}

auto qn_eloc_envs_logpsi(
    const QnpepsElocConfig& cfg,
    const cf* peps,
    const u8* samples,
    i64 n_samples,
    f64* logpsi_out,
    Linalg& linalg
) -> void;

auto run_cache_execute(
    qn_eloc::env::RunCache& state,
    const cf* peps,
    const u8* samples,
    f64* logpsi_out,
    f64* e_loc_out,
    cf* o_rows_dev,
    cf* o_rows_host,
    cf* T_dev,
    f64 lambda
) -> void;

struct qnpeps_eloc_ctx
{
    explicit qnpeps_eloc_ctx(std::unique_ptr<qnpeps::Session> incoming)
        : session(std::move(incoming)), state(session->linalg())
    {
    }

    std::unique_ptr<qnpeps::Session> session{};
    qn_eloc::env::RunCache state;
};

auto qn_eloc_run_impl(
    const QnpepsElocConfig& cfg,
    const cf* peps,
    const u8* samples,
    i64 n_samples,
    const QnpepsElocTermTable* tt,
    f64* logpsi_out,
    f64* e_loc_out,
    cf* o_rows_dev,
    cf* o_rows_host,
    cf* T_dev,
    f64 lambda,
    Linalg& linalg
) -> void;

auto qn_eloc_ctx_create_impl(
    const QnpepsElocConfig& cfg,
    i64 n_samples,
    const QnpepsElocTermTable& terms,
    uint32_t flags,
    std::unique_ptr<qnpeps::Session> session,
    qnpeps_eloc_ctx** out
) -> void;

auto qn_eloc_ctx_destroy_impl(qnpeps_eloc_ctx* ctx) -> void;

auto qn_eloc_ctx_run_async_impl(
    qnpeps_eloc_ctx& ctx,
    const cf* peps,
    const u8* samples,
    f64* logpsi_out,
    f64* e_loc_out,
    cf* o_rows_dev,
    cf* o_rows_host,
    cf* T_dev,
    f64 lambda
) -> void;

auto qn_eloc_ctx_run_impl(
    qnpeps_eloc_ctx& ctx,
    const cf* peps,
    const u8* samples,
    f64* logpsi_out,
    f64* e_loc_out,
    cf* o_rows_dev,
    cf* o_rows_host,
    cf* T_dev,
    f64 lambda
) -> void;

auto qn_eloc_gram_tile_impl(
    const QnpepsElocConfig& cfg,
    const cf* rows_a,
    const u8* samples_a,
    i64 ns_a,
    const cf* rows_b,
    const u8* samples_b,
    i64 ns_b,
    cf* tile_out,
    cudaStream_t stream
) -> void;

auto qn_eloc_compact_count(const QnpepsElocConfig& cfg) -> i64;
