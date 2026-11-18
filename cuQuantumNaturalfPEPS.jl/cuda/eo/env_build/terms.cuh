#ifndef QNPEPS_ELOC_ENV_BUILD_TERMS_CUH
#define QNPEPS_ELOC_ENV_BUILD_TERMS_CUH

#include "common.cuh"
#include "core/arena_cursor.cuh"
#include "dans_qnpeps_eloc.h"
#include "term_validation.cuh"

#include <cuComplex.h>
#include <string>
#include <vector>

namespace qn_eloc::env
{
using qnpeps::operator""_uz;

inline constexpr int k_keep_position{-1};
inline constexpr usize k_term_sites{QNPEPS_ELOC_TERM_MAX_SITES};

enum class Bucket
{
    horizontal,
    fourbody,
    longer_horizontal,
    slab
};

struct CuLocalState
{
    int site_count{};
    int dim_phys{};
    qnpeps::CuArray<int, k_term_sites> site{};
};

[[nodiscard]] __host__ __device__ inline auto local_state_index(
    const CuLocalState& local_state, const u8* lane_sample
) noexcept -> int
{
    auto state = 0;
    for (auto k = 0_uz; k < static_cast<usize>(local_state.site_count); ++k)
        state = state * local_state.dim_phys + lane_sample[local_state.site[k]];
    return state;
}

struct TermPiece
{
    Bucket bucket{};
    int pass{};
    int row_count{};
    int active_slot{-1};
    CuLocalState local_state{};
    int moving_count{};
    qnpeps::CuArray<int, k_term_sites> moving{};
    qnpeps::CuArray<bool, k_term_sites> is_moving{};
    bool is_one_hot{};
    i64 state_offset{};
    i64 weight_offset{};
    i64 target_offset{};
    const u8* device_active{};
    const cuDoubleComplex* device_coefficients{};
    const cf* device_weights{};
    const u8* device_targets{};
};

struct CuDiagonalTerm
{
    CuLocalState local_state{};
    i64 value_offset{};
};

struct TermPlan
{
    std::vector<CuDiagonalTerm> diagonal{};
    std::vector<cuDoubleComplex> diagonal_values{};
    std::vector<TermPiece> original{};
    std::vector<TermPiece> transposed{};
    std::vector<u8> active{};
    std::vector<cuDoubleComplex> coefficients{};
    std::vector<cf> weights{};
    std::vector<u8> targets{};
};

struct DeviceTermPools
{
    CuDiagonalTerm* diagonal{};
    cuDoubleComplex* diagonal_values{};
    u8* active{};
    cuDoubleComplex* coefficients{};
    cf* weights{};
    u8* targets{};
};

struct CuActiveSlot
{
    CuLocalState local_state{};
    const u8* active{};
};

struct ActiveLists
{
    int slots{};
    int max_lanes{};
    CuActiveSlot* device_slots{};
    int* counts{};
    int* indices{};
    std::vector<int> host_counts{};
};

struct CuProjectGatherArgs
{
    const cf* site{};
    const u8* samples{};
    i64 lane_stride{};
    const int* active_indices{};
    int site_index{};
    i64 slab{};
    CuLocalState local_state{};
    int support_position{k_keep_position};
    const u8* targets{};
    cf* out{};
    i64 out_stride{};
    int active{};
};

struct CuProjectWeightedArgs
{
    const cf* site{};
    const u8* samples{};
    i64 lane_stride{};
    const int* active_indices{};
    i64 slab{};
    CuLocalState local_state{};
    int support_position{};
    const cf* weights{};
    cf* out{};
    i64 out_stride{};
    int active{};
};

struct CuDiagonalArgs
{
    f64* e_loc{};
    const u8* samples{};
    i64 lane_stride{};
    const CuDiagonalTerm* terms{};
    const cuDoubleComplex* values{};
    int term_count{};
    int lanes{};
};

struct CuReduceTermArgs
{
    f64* e_loc{};
    const cf* value{};
    const f64* f_top{};
    const f64* f_down{};
    const f64* f_rail_l{};
    const f64* f_rail_r{};
    const f64* f_chain{};
    const f64* logpsi{};
    const int* active_indices{};
    const u8* samples{};
    i64 lane_stride{};
    CuLocalState local_state{};
    const cuDoubleComplex* coefficients{};
    int active{};
};

struct CuActiveListArgs
{
    const u8* samples{};
    i64 lane_stride{};
    const CuActiveSlot* slots{};
    int slot_count{};
    int lanes{};
    int max_lanes{};
    int* indices{};
    int* counts{};
};

__global__ auto cu_compact_lanes(
    cf* out,
    i64 out_stride,
    const cf* in,
    i64 in_stride,
    const int* active_indices,
    i64 elems,
    int active
) -> void;

__global__ auto cu_project_site_gather(CuProjectGatherArgs args) -> void;

__global__ auto cu_project_site_weighted(CuProjectWeightedArgs args) -> void;

__global__ auto cu_diagonal(CuDiagonalArgs args) -> void;

__global__ auto cu_reduce_term_indexed(CuReduceTermArgs args) -> void;

__global__ auto cu_build_active_lists(CuActiveListArgs args) -> void;

struct MovingBounds
{
    int top_row{};
    int bottom_row{};
    int left_column{};
    int right_column{};
};

[[nodiscard]] auto support_position(const TermPiece& piece, int site) noexcept -> int;

[[nodiscard]] auto moving_bounds(const TermPiece& piece, int ly) noexcept -> MovingBounds;

[[nodiscard]] auto classify_route(const QnpepsElocConfig& cfg, TermPiece& piece) -> bool;

[[nodiscard]] auto slab_rows(const std::vector<TermPiece>& pieces) noexcept -> int;

[[nodiscard]] auto piece_sites_text(const QnpepsElocConfig& cfg, const TermPiece& piece)
    -> std::string;

[[nodiscard]] auto plan_terms(
    const QnpepsElocConfig& cfg, const QnpepsElocTermTable& table, TermPlan& plan
) -> bool;

auto bind_device_pools(std::vector<TermPiece>& pieces, const DeviceTermPools& pools) -> void;

auto active_lists_setup(
    ActiveLists& lists,
    const std::vector<TermPiece>& pieces,
    int max_lanes,
    qnpeps::ArenaCursor& context_arena
) -> void;

auto upload_active_slots(
    ActiveLists& lists,
    const std::vector<TermPiece>& pieces,
    const u8* device_active,
    qnpeps::ArenaCursor& context_arena
) -> void;

auto prepare_active_lists(
    ActiveLists& lists, const u8* samples, i64 lane_stride, int lanes, cudaStream_t stream
) -> void;
}

#endif
