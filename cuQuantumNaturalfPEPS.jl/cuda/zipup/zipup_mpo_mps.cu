#include "core/arena_cursor.cuh"
#include "core/cuda_utils.cuh"
#include "core/defer.cuh"
#include "core/session.cuh"
#include "linalg/transfer.cuh"
#include "sampler/trunc_svd.cuh"
#include "zipup/zipup_mpo_mps.cuh"
#include "zipup/kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda/std/cmath>
#include <limits>
#include <span>
#include <vector>

namespace qnpeps::zipup
{
namespace
{
inline constexpr usize k_mpo_axis_count{4};
inline constexpr usize k_mps_axis_count{3};
inline constexpr usize k_mpo_left{0};
inline constexpr usize k_mpo_physical_in{1};
inline constexpr usize k_mpo_physical_out{2};
inline constexpr usize k_mpo_right{3};
inline constexpr usize k_mps_left{0};
inline constexpr usize k_mps_physical{1};
inline constexpr usize k_mps_right{2};

struct SiteLayout
{
    Shape mpo{};
    Shape mps{};
    Shape output{};
    usize mpo_offset{};
    usize mps_offset{};
    usize output_offset{};
};

struct Layout
{
    std::vector<SiteLayout> sites{};
    usize mpo_elements{};
    usize mps_elements{};
    usize output_elements{};
};

[[nodiscard]] auto checked_multiply(usize left, usize right, usize& output) -> bool
{
    if (right != 0 and left > std::numeric_limits<usize>::max() / right) return false;
    output = left * right;
    return true;
}

[[nodiscard]] auto checked_add(usize left, usize right, usize& output) -> bool
{
    if (left > std::numeric_limits<usize>::max() - right) return false;
    output = left + right;
    return true;
}

[[nodiscard]] auto product_fits_int(std::initializer_list<int> values) -> bool
{
    auto product = 1;
    for (const auto value : values)
    {
        if (value < 1 or product > std::numeric_limits<int>::max() / value) return false;
        product *= value;
    }
    return true;
}

[[nodiscard]] auto element_product_fits_bytes(std::initializer_list<int> values) -> bool
{
    usize elements{1};
    for (const auto value : values)
    {
        if (value < 1 or not checked_multiply(elements, static_cast<usize>(value), elements))
            return false;
    }
    usize bytes{};
    return checked_multiply(elements, sizeof(cuFloatComplex), bytes)
           and bytes <= static_cast<usize>(std::numeric_limits<i64>::max());
}

[[nodiscard]] auto append_elements(usize& total, const Shape& shape) -> bool
{
    usize next{};
    if (not checked_add(total, shape.num_elems(), next)) return false;
    total = next;
    return true;
}

[[nodiscard]] auto validate_descriptor(const QnpepsZipupMpoMpsDesc* descriptor, Layout& layout)
    -> qnpeps_status
{
    if (not descriptor) return QNPEPS_ERR_NULL_ARG;
    if (descriptor->struct_size != sizeof(QnpepsZipupMpoMpsDesc)) return QNPEPS_ERR_BAD_VERSION;
    if (descriptor->reserved != 0 or descriptor->num_sites < 1 or descriptor->maxdim < 1)
        return QNPEPS_ERR_BAD_CONFIG;
    if (not descriptor->mpo_dims or not descriptor->mps_dims) return QNPEPS_ERR_NULL_ARG;

    const auto num_sites = static_cast<usize>(descriptor->num_sites);
    layout = {};
    layout.sites.reserve(num_sites);
    auto output_left = 1;
    auto previous_mpo_right = 1;
    auto previous_mps_right = 1;

    for (auto site = 0_uz; site < num_sites; ++site)
    {
        const auto mpo_base = site * k_mpo_axis_count;
        const auto mps_base = site * k_mps_axis_count;
        const int mpo_left{descriptor->mpo_dims[mpo_base + k_mpo_left]};
        const int physical_in{descriptor->mpo_dims[mpo_base + k_mpo_physical_in]};
        const int physical_out{descriptor->mpo_dims[mpo_base + k_mpo_physical_out]};
        const int mpo_right{descriptor->mpo_dims[mpo_base + k_mpo_right]};
        const int mps_left{descriptor->mps_dims[mps_base + k_mps_left]};
        const int mps_physical{descriptor->mps_dims[mps_base + k_mps_physical]};
        const int mps_right{descriptor->mps_dims[mps_base + k_mps_right]};
        const auto positive = mpo_left > 0 and physical_in > 0 and physical_out > 0
                              and mpo_right > 0 and mps_left > 0 and mps_physical > 0
                              and mps_right > 0;
        if (not positive) return QNPEPS_ERR_BAD_CONFIG;
        if (mpo_left != previous_mpo_right or mps_left != previous_mps_right)
            return QNPEPS_ERR_BAD_CONFIG;
        if (physical_in != mps_physical) return QNPEPS_ERR_BAD_CONFIG;

        const auto valid_integer_products = product_fits_int({output_left, physical_out})
                                            and product_fits_int({mps_right, mpo_right})
                                            and product_fits_int({output_left, mpo_left})
                                            and product_fits_int({physical_in, mps_right})
                                            and product_fits_int({mpo_left, physical_in})
                                            and product_fits_int({physical_out, mpo_right});
        if (not valid_integer_products)
        {
            return QNPEPS_ERR_BAD_CONFIG;
        }

        const int rows{output_left * physical_out};
        const int cols{mps_right * mpo_right};
        const int output_right{std::min({descriptor->maxdim, rows, cols})};
        const auto valid_byte_products =
            element_product_fits_bytes({mpo_left, physical_in, physical_out, mpo_right})
            and element_product_fits_bytes({mps_left, mps_physical, mps_right})
            and element_product_fits_bytes({output_left, physical_out, output_right})
            and element_product_fits_bytes({output_left, mpo_left, mps_left})
            and element_product_fits_bytes({output_left, mpo_left, physical_in, mps_right})
            and element_product_fits_bytes({output_left, mps_right, physical_out, mpo_right});
        if (not valid_byte_products)
        {
            return QNPEPS_ERR_BAD_CONFIG;
        }
        SiteLayout site_layout{
            .mpo = Shape{mpo_left, physical_in, physical_out, mpo_right},
            .mps = Shape{mps_left, mps_physical, mps_right},
            .output = Shape{output_left, physical_out, output_right},
            .mpo_offset = layout.mpo_elements,
            .mps_offset = layout.mps_elements,
            .output_offset = layout.output_elements,
        };
        const auto appended_mpo = append_elements(layout.mpo_elements, site_layout.mpo);
        const auto appended_mps = append_elements(layout.mps_elements, site_layout.mps);
        const auto appended_output = append_elements(layout.output_elements, site_layout.output);
        if (not appended_mpo or not appended_mps or not appended_output)
        {
            return QNPEPS_ERR_BAD_CONFIG;
        }
        layout.sites.push_back(std::move(site_layout));
        output_left = output_right;
        previous_mpo_right = mpo_right;
        previous_mps_right = mps_right;
    }

    if (previous_mpo_right != 1 or previous_mps_right != 1) return QNPEPS_ERR_BAD_CONFIG;
    return QNPEPS_OK;
}

[[nodiscard]] auto elements_to_bytes(usize elements, usize& bytes) -> bool
{
    return checked_multiply(elements, sizeof(cuFloatComplex), bytes)
           and bytes <= static_cast<usize>(std::numeric_limits<i64>::max());
}

[[nodiscard]] auto validate_args(const Layout& layout, const QnpepsZipupMpoMpsArgs* args)
    -> qnpeps_status
{
    if (not args) return QNPEPS_ERR_NULL_ARG;
    if (args->struct_size != sizeof(QnpepsZipupMpoMpsArgs)) return QNPEPS_ERR_BAD_VERSION;
    if (args->reserved != 0) return QNPEPS_ERR_BAD_CONFIG;
    if (not args->mpo or not args->mps or not args->output or not args->log_gauge)
        return QNPEPS_ERR_NULL_ARG;

    usize mpo_bytes{};
    usize mps_bytes{};
    usize output_bytes_required{};
    const auto valid_mpo_bytes = elements_to_bytes(layout.mpo_elements, mpo_bytes);
    const auto valid_mps_bytes = elements_to_bytes(layout.mps_elements, mps_bytes);
    const auto valid_output_bytes =
        elements_to_bytes(layout.output_elements, output_bytes_required);
    if (not valid_mpo_bytes or not valid_mps_bytes or not valid_output_bytes)
    {
        return QNPEPS_ERR_BAD_CONFIG;
    }
    const auto valid_input_bytes = args->mpo_bytes == mpo_bytes and args->mps_bytes == mps_bytes;
    const auto valid_output_capacity = args->output_bytes >= output_bytes_required;
    if (not valid_input_bytes or not valid_output_capacity)
    {
        return QNPEPS_ERR_BAD_CONFIG;
    }
    return QNPEPS_OK;
}

class MaterializedPanelProvider
{
  public:
    MaterializedPanelProvider(
        const std::vector<DeviceTensor>& mpo, const std::vector<DeviceTensor>& mps
    )
        : mpo_(mpo), mps_(mps)
    {
    }

    [[nodiscard]] auto make_panel(
        usize site, const DeviceTensor& carried_factor, ArenaCursor& scratch, Linalg& la
    ) const -> DeviceTensor
    {
        DeviceTensor left_mps{};
        const auto left_ok = contract(
            scratch,
            la,
            {
                .dims_a = carried_factor.dim,
                .contracted_a = {2},
                .dims_b = mps_[site].dim,
                .contracted_b = {0},
            },
            carried_factor,
            mps_[site],
            left_mps
        );
        if (not left_ok) return {};

        DeviceTensor unpermuted_panel{};
        const auto panel_ok = contract(
            scratch,
            la,
            {
                .dims_a = left_mps.dim,
                .contracted_a = {1, 2},
                .dims_b = mpo_[site].dim,
                .contracted_b = {0, 1},
            },
            left_mps,
            mpo_[site],
            unpermuted_panel
        );
        if (not panel_ok) return {};
        return permute_axes(scratch, unpermuted_panel, {0, 2, 1, 3}, false, la.stream());
    }

  private:
    const std::vector<DeviceTensor>& mpo_;
    const std::vector<DeviceTensor>& mps_;
};

class FusedPepsPanelProvider
{
  public:
    FusedPepsPanelProvider(
        const std::vector<DeviceTensor>& row_ket,
        const std::vector<DeviceTensor>* environment,
        DeviceTensor unit_environment
    )
        : row_ket_(row_ket), environment_(environment), unit_environment_(unit_environment)
    {
    }

    [[nodiscard]] auto make_panel(
        usize site, const DeviceTensor& grouped_carried_factor, ArenaCursor& scratch, Linalg& la
    ) const -> DeviceTensor
    {
        const auto& ket = row_ket_[site];
        const auto& environment = environment_ ? (*environment_)[site] : unit_environment_;
        const auto ket_left = ket.dim[4];
        const auto bra_left = ket.dim[4];
        const DeviceTensor carried_factor{
            {grouped_carried_factor.dim[0], ket_left, bra_left, grouped_carried_factor.dim[2]},
            grouped_carried_factor.d
        };

        DeviceTensor left_environment{};
        const auto environment_ok = contract(
            scratch,
            la,
            {
                .dims_a = carried_factor.dim,
                .contracted_a = {3},
                .dims_b = environment.dim,
                .contracted_b = {0},
            },
            carried_factor,
            environment,
            left_environment
        );
        if (not environment_ok) return {};

        DeviceTensor left_environment_ket{};
        const auto ket_ok = contract(
            scratch,
            la,
            {
                .dims_a = left_environment.dim,
                .contracted_a = {1, 3},
                .dims_b = ket.dim,
                .contracted_b = {4, 3},
            },
            left_environment,
            ket,
            left_environment_ket
        );
        if (not ket_ok) return {};

        DeviceTensor column_tensor{};
        const auto bra_ok = contract(
            scratch,
            la,
            {
                .dims_a = left_environment_ket.dim,
                .contracted_a = {1, 2, 4},
                .dims_b = ket.dim,
                .contracted_b = {4, 3, 0},
                .transforms = {.conj_b = true},
            },
            left_environment_ket,
            ket,
            column_tensor
        );
        if (not bra_ok) return {};

        const auto column_matrix =
            permute_axes(scratch, column_tensor, {0, 2, 4, 1, 3, 5}, false, la.stream());
        return DeviceTensor{
            {
                column_tensor.dim[0],
                column_tensor.dim[2] * column_tensor.dim[4],
                column_tensor.dim[1],
                column_tensor.dim[3] * column_tensor.dim[5],
            },
            column_matrix.d
        };
    }

  private:
    const std::vector<DeviceTensor>& row_ket_;
    const std::vector<DeviceTensor>* environment_{};
    DeviceTensor unit_environment_{};
};

}

auto rangefinder(State& state, Linalg& la, const Arenas& arenas, const RangefinderArgs& args)
    -> void
{
    const auto route = trunc_svd::require_dlenv_route();
    if (route == trunc_svd::Route::invalid) return;

    int rank{std::min({args.maxdim, args.rows, args.cols})};
    if (rank < 1) rank = 1;
    *args.rank_out = rank;
    const auto rank_u = static_cast<usize>(rank);
    const CuMatrixCF32 input_matrix{args.input.d, args.rows, args.cols};

    auto sketch = alloc(arenas.scratch, {args.rows, rank});
    auto projection = alloc(arenas.scratch, {args.cols, rank});
    if (err_state() != QNPEPS_OK) return;

    const std::pair key{args.cols, rank};
    auto omega_it = state.omegas->find(key);
    if (omega_it == state.omegas->end())
    {
        std::vector<cuFloatComplex> host_omega{};
        host_omega.resize(static_cast<usize>(args.cols) * rank_u);
        state.rangefinder_rng->fill_complex_normal(std::span{host_omega});
        cuFloatComplex* device_omega{};
        CUDA_CHECK(cudaMalloc(&device_omega, host_omega.size() * sizeof(cuFloatComplex)));
        if (err_state() != QNPEPS_OK) return;
        upload(device_omega, host_omega.data(), host_omega.size());
        if (err_state() != QNPEPS_OK)
        {
            CUDA_NOCHECK(cudaFree(device_omega));
            return;
        }
        omega_it = state.omegas->emplace(key, device_omega).first;
    }

    const CuMatrixCF32 omega_matrix{omega_it->second, args.cols, rank};
    const CuMatrixCF32 sketch_matrix{sketch.d, args.rows, rank};
    const CuMatrixCF32 projection_matrix{projection.d, args.cols, rank};
    la.matmul(input_matrix, omega_matrix, sketch_matrix);
    for (auto iteration = 0; iteration < 2; ++iteration)
    {
        la.matmul_left_adj(input_matrix, sketch_matrix, projection_matrix);
        la.matmul(input_matrix, projection_matrix, sketch_matrix);
    }

    if (route == trunc_svd::Route::svd)
    {
        trunc_svd::orthonormalize_panel(la, sketch_matrix);
        if (err_state() != QNPEPS_OK) return;
    }
    else
    {
        const auto qr_layout = la.qr_scratch(sketch_matrix);
        auto qr_scratch{arenas.scratch.take<char>(qr_layout.total())};
        la.qr(sketch_matrix, qr_scratch, qr_layout);
        if (state.fail_flag)
        {
            const auto* qr_status = byte_offset<int>(qr_scratch, qr_layout.reflector_bytes);
            cu_or_status<<<1, 1, 0, la.stream()>>>(qr_status, state.fail_flag);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    *args.q = alloc(arenas.known, {args.rows, rank});
    const auto q_count = static_cast<usize>(args.rows) * rank_u;
    copy_device_async(la, args.q->d, sketch.d, q_count);
    *args.r = alloc(arenas.scratch, {rank, args.cols});
    la.matmul_left_adj(sketch_matrix, input_matrix, CuMatrixCF32{args.r->d, rank, args.cols});
}

auto normalize_factor(Linalg& la, ArenaCursor& scratch, DeviceTensor factor, f64* device_scale)
    -> void
{
    const auto elements = static_cast<i64>(factor.num_elems());
    auto* device_inverse_scale = scratch.take<f32>(1);
    const AbsmaxInverseArgs args{factor.d, elements, device_inverse_scale, device_scale};
    cu_absmax_inverse<<<1, k_tree_reduce_threads, 0, la.stream()>>>(args);
    CUDA_CHECK(cudaGetLastError());
    const auto blocks = static_cast<u32>(ceil_div(elements, k_tree_reduce_threads));
    cu_apply_inverse_scale<<<blocks, k_tree_reduce_threads, 0, la.stream()>>>(
        factor.d, elements, device_inverse_scale
    );
    CUDA_CHECK(cudaGetLastError());
}

auto accumulate_log_scales(const State& state, usize count, f64& log_scale) -> void
{
    std::vector<f64> scales{};
    scales.resize(count);
    download(scales.data(), state.device_scales, count);
    for (const auto scale : scales)
    {
        if (not std::isfinite(scale))
        {
            set_err(QNPEPS_ERR_INTERNAL);
            return;
        }
        if (scale > 0.0) log_scale += std::log(scale);
    }
}

auto release_omegas(State& state) -> void
{
    if (not state.omegas) return;
    for (auto& entry : *state.omegas)
        maybe_free_device(entry.second);
    state.omegas->clear();
}

auto fused_peps_row(State& state, Linalg& la, const Arenas& arenas, const FusedPepsRowArgs& args)
    -> std::vector<DeviceTensor>
{
    if (not args.row_ket or args.row_ket->empty())
    {
        set_err(QNPEPS_ERR_INTERNAL);
        return {};
    }
    FusedPepsPanelProvider provider{*args.row_ket, args.environment, args.unit_environment};
    return sweep(
        state,
        la,
        arenas,
        provider,
        {
            .num_sites = args.row_ket->size(),
            .maxdim = args.maxdim,
        }
    );
}

auto output_bytes(const QnpepsZipupMpoMpsDesc* descriptor) -> i64
{
    Layout layout{};
    if (validate_descriptor(descriptor, layout) != QNPEPS_OK) return -1;
    usize bytes{};
    if (not elements_to_bytes(layout.output_elements, bytes)) return -1;
    return static_cast<i64>(bytes);
}

auto execute(const QnpepsZipupMpoMpsDesc* descriptor, const QnpepsZipupMpoMpsArgs* args)
    -> qnpeps_status
{
    Layout layout{};
    const auto descriptor_status = validate_descriptor(descriptor, layout);
    if (descriptor_status != QNPEPS_OK) return set_err(descriptor_status);
    const auto args_status = validate_args(layout, args);
    if (args_status != QNPEPS_OK) return set_err(args_status);

    auto session = make_session(static_cast<cudaStream_t>(args->stream));
    if (not session) return err_state();
    auto& linalg = session->linalg();
    const auto stream = linalg.stream();
    auto transient_arena = linalg.transient_arena();
    auto& carved = transient_arena.cursor();
    auto* fail_flag = carved.take<int>(1);
    auto* device_scales = carved.take<f64>(layout.sites.size());
    auto* initial_factor = carved.take<cuFloatComplex>(1);
    const auto region = carved.remaining() / 3;
    auto known = carved.take_subarena(region);
    auto rolling_r = carved.take_subarena(region);
    auto scratch = carved.take_subarena(carved.remaining());

    std::map<std::pair<int, int>, cuFloatComplex*> omegas{};
    RangefinderRng rangefinder_rng{k_rangefinder_seed};
    State state{
        .initial_factor = initial_factor,
        .device_scales = device_scales,
        .fail_flag = fail_flag,
        .omegas = &omegas,
        .rangefinder_rng = &rangefinder_rng,
    };
    DEFER([&] { release_omegas(state); });

    constexpr cuFloatComplex one{1.0f, 0.0f};
    upload_async(linalg, initial_factor, &one, 1);
    zero_async(linalg, fail_flag, 1);
    if (err_state() != QNPEPS_OK) return err_state();

    std::vector<DeviceTensor> mpo{};
    std::vector<DeviceTensor> mps{};
    mpo.reserve(layout.sites.size());
    mps.reserve(layout.sites.size());
    const auto* mpo_base = static_cast<const cuFloatComplex*>(args->mpo);
    const auto* mps_base = static_cast<const cuFloatComplex*>(args->mps);
    for (const auto& site : layout.sites)
    {
        mpo.push_back(
            DeviceTensor{site.mpo, const_cast<cuFloatComplex*>(mpo_base + site.mpo_offset)}
        );
        mps.push_back(
            DeviceTensor{site.mps, const_cast<cuFloatComplex*>(mps_base + site.mps_offset)}
        );
    }

    MaterializedPanelProvider provider{mpo, mps};
    f64 log_scale{};
    const Arenas arenas{known, rolling_r, scratch};
    auto output = sweep(
        state,
        linalg,
        arenas,
        provider,
        {
            .num_sites = layout.sites.size(),
            .maxdim = descriptor->maxdim,
        }
    );
    if (err_state() == QNPEPS_OK)
    {
        auto* output_base = static_cast<cuFloatComplex*>(args->output);
        for (auto site = 0_uz; site < layout.sites.size(); ++site)
        {
            copy_device_async(
                linalg,
                output_base + layout.sites[site].output_offset,
                output[site].d,
                output[site].num_elems()
            );
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    int fail_host{};
    if (err_state() == QNPEPS_OK) download(&fail_host, fail_flag, 1);
    if (fail_host != 0) set_err(QNPEPS_ERR_CUDA);
    if (err_state() == QNPEPS_OK) accumulate_log_scales(state, layout.sites.size(), log_scale);
    if (err_state() == QNPEPS_OK) *args->log_gauge = log_scale;
    return err_state();
}
}
