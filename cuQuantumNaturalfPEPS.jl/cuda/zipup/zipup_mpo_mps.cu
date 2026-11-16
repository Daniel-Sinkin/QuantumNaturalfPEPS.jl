#include "capi/internal/format.cuh"
#include "core/arena_cursor.cuh"
#include "core/cuda_utils.cuh"
#include "core/defer.cuh"
#include "core/session.cuh"
#include "linalg/transfer.cuh"
#include "sampler/trunc_svd.cuh"
#include "zipup/power_scaling.cuh"
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

auto validate_descriptor(const QnpepsZipupMpoMpsDesc* descriptor, Layout& layout) -> void
{
    require(descriptor != nullptr, Status::null_argument, "zipup descriptor is null");
    require_struct_size(*descriptor, "zipup descriptor");
    const auto valid_counts =
        descriptor->reserved == 0 and descriptor->num_sites >= 1 and descriptor->maxdim >= 1;
    require(valid_counts, Status::bad_config, "zipup descriptor rejects {}", *descriptor);
    const auto has_dims = descriptor->mpo_dims != nullptr and descriptor->mps_dims != nullptr;
    require(has_dims, Status::null_argument, "zipup descriptor misses the MPO or MPS dimensions");

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
        require(positive, Status::bad_config, "zipup site {} has a nonpositive extent", site);
        const auto bonds_chain = mpo_left == previous_mpo_right and mps_left == previous_mps_right;
        require(
            bonds_chain,
            Status::bad_config,
            "zipup site {} left bonds MPO {} and MPS {} do not match the previous right bonds {} "
            "and {}",
            site,
            mpo_left,
            mps_left,
            previous_mpo_right,
            previous_mps_right
        );
        require(
            physical_in == mps_physical,
            Status::bad_config,
            "zipup site {} MPO input dimension {} differs from MPS physical dimension {}",
            site,
            physical_in,
            mps_physical
        );

        const auto valid_integer_products = product_fits_int({output_left, physical_out})
                                            and product_fits_int({mps_right, mpo_right})
                                            and product_fits_int({output_left, mpo_left})
                                            and product_fits_int({physical_in, mps_right})
                                            and product_fits_int({mpo_left, physical_in})
                                            and product_fits_int({physical_out, mpo_right});
        require(
            valid_integer_products,
            Status::bad_config,
            "zipup site {} matrix extents exceed the int range",
            site
        );

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
        require(
            valid_byte_products,
            Status::bad_config,
            "zipup site {} tensor bytes exceed the i64 range",
            site
        );
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
        require(
            appended_mpo and appended_mps and appended_output,
            Status::bad_config,
            "zipup element totals overflow at site {}",
            site
        );
        layout.sites.push_back(std::move(site_layout));
        output_left = output_right;
        previous_mpo_right = mpo_right;
        previous_mps_right = mps_right;
    }

    require(
        previous_mpo_right == 1 and previous_mps_right == 1,
        Status::bad_config,
        "zipup chain ends in right bonds MPO {} and MPS {}, expected 1 and 1",
        previous_mpo_right,
        previous_mps_right
    );
}

[[nodiscard]] auto elements_to_bytes(usize elements, usize& bytes) -> bool
{
    return checked_multiply(elements, sizeof(cuFloatComplex), bytes)
           and bytes <= static_cast<usize>(std::numeric_limits<i64>::max());
}

auto validate_args(const Layout& layout, const QnpepsZipupMpoMpsArgs* args) -> void
{
    require(args != nullptr, Status::null_argument, "zipup arguments are null");
    require_struct_size(*args, "zipup arguments");
    require(
        args->reserved == 0,
        Status::bad_config,
        "zipup arguments reserved field is {}",
        args->reserved
    );
    const auto has_buffers = args->mpo and args->mps and args->output and args->log_gauge;
    require(has_buffers, Status::null_argument, "zipup arguments miss a buffer");

    usize mpo_bytes{};
    usize mps_bytes{};
    usize output_bytes_required{};
    const auto valid_mpo_bytes = elements_to_bytes(layout.mpo_elements, mpo_bytes);
    const auto valid_mps_bytes = elements_to_bytes(layout.mps_elements, mps_bytes);
    const auto valid_output_bytes =
        elements_to_bytes(layout.output_elements, output_bytes_required);
    require(
        valid_mpo_bytes and valid_mps_bytes and valid_output_bytes,
        Status::bad_config,
        "zipup byte totals exceed the i64 range"
    );
    const auto valid_input_bytes = args->mpo_bytes == mpo_bytes and args->mps_bytes == mps_bytes;
    require(
        valid_input_bytes,
        Status::bad_config,
        "zipup inputs hold {} MPO and {} MPS bytes, the descriptor needs {} and {}",
        args->mpo_bytes,
        args->mps_bytes,
        mpo_bytes,
        mps_bytes
    );
    require(
        args->output_bytes >= output_bytes_required,
        Status::bad_config,
        "zipup output holds {} bytes, needs {}",
        args->output_bytes,
        output_bytes_required
    );
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
        contract(
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

        DeviceTensor unpermuted_panel{};
        contract(
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
        contract(
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

        DeviceTensor left_environment_ket{};
        contract(
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

        DeviceTensor column_tensor{};
        contract(
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

auto analyze_power_input(CuMatrixCF32 input, int* exponent, int* fail_flag, cudaStream_t stream)
    -> void
{
    cu_input_exponent<<<1, k_tree_reduce_threads, 0, stream>>>(
        {input.data(), static_cast<i64>(input.rows()) * input.cols(), exponent, fail_flag}
    );
    check(cudaGetLastError());
}

auto scale_power_intermediate(
    CuMatrixCF32 matrix,
    const int* input_exponent,
    int next_inner_dimension,
    int* fail_flag,
    cudaStream_t stream
) -> void
{
    cu_scale_power_intermediate<<<1, k_tree_reduce_threads, 0, stream>>>(
        {matrix.data(),
         static_cast<i64>(matrix.rows()) * matrix.cols(),
         input_exponent,
         next_inner_dimension,
         fail_flag}
    );
    check(cudaGetLastError());
}

}

auto raise_rangefinder_failure(int failure) -> void
{
    require(
        (failure & k_power_nonfinite_flag) == 0,
        "nonfinite zipup power intermediate, failure flags {}",
        failure
    );
    require(failure == 0, Status::cuda, "zipup rangefinder QR reported failure flags {}", failure);
}

auto rangefinder(State& state, Linalg& la, const Arenas& arenas, const RangefinderArgs& args)
    -> void
{
    int rank{std::min({args.maxdim, args.rows, args.cols})};
    if (rank < 1) rank = 1;
    *args.rank_out = rank;
    const auto rank_u = static_cast<usize>(rank);
    const CuMatrixCF32 input_matrix{args.input.d, args.rows, args.cols};

    auto sketch = alloc(arenas.scratch, {args.rows, rank});
    auto projection = alloc(arenas.scratch, {args.cols, rank});
    require(
        state.power_input_exponent != nullptr, "zipup rangefinder has no power exponent buffer"
    );

    const std::pair key{args.cols, rank};
    auto omega_it = state.omegas->find(key);
    if (omega_it == state.omegas->end())
    {
        std::vector<cuFloatComplex> host_omega{};
        host_omega.resize(static_cast<usize>(args.cols) * rank_u);
        state.rangefinder_rng->fill_complex_normal(std::span{host_omega});
        cuFloatComplex* device_omega{};
        check(cudaMalloc(&device_omega, host_omega.size() * sizeof(cuFloatComplex)));
        auto stored = false;
        const Defer free_after_failure{[&]
                                       {
                                           if (not stored) unchecked(cudaFree(device_omega));
                                       }};
        upload(device_omega, host_omega.data(), host_omega.size());
        omega_it = state.omegas->emplace(key, device_omega).first;
        stored = true;
    }

    const CuMatrixCF32 omega_matrix{omega_it->second, args.cols, rank};
    const CuMatrixCF32 sketch_matrix{sketch.d, args.rows, rank};
    const CuMatrixCF32 projection_matrix{projection.d, args.cols, rank};
    analyze_power_input(input_matrix, state.power_input_exponent, state.fail_flag, la.stream());
    la.matmul(input_matrix, omega_matrix, sketch_matrix);
    scale_power_intermediate(
        sketch_matrix, state.power_input_exponent, args.rows, state.fail_flag, la.stream()
    );
    for (auto iteration = 0; iteration < 2; ++iteration)
    {
        la.matmul_left_adj(input_matrix, sketch_matrix, projection_matrix);
        scale_power_intermediate(
            projection_matrix, state.power_input_exponent, args.cols, state.fail_flag, la.stream()
        );
        la.matmul(input_matrix, projection_matrix, sketch_matrix);
        const auto next_inner_dimension = iteration == 0 ? args.rows : 0;
        scale_power_intermediate(
            sketch_matrix,
            state.power_input_exponent,
            next_inner_dimension,
            state.fail_flag,
            la.stream()
        );
    }

    if (state.svd_route)
    {
        trunc_svd::orthonormalize_panel(la, sketch_matrix);
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
            check(cudaGetLastError());
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
    check(cudaGetLastError());
    const auto blocks = static_cast<u32>(ceil_div(elements, k_tree_reduce_threads));
    cu_apply_inverse_scale<<<blocks, k_tree_reduce_threads, 0, la.stream()>>>(
        factor.d, elements, device_inverse_scale
    );
    check(cudaGetLastError());
}

auto accumulate_log_scales(const State& state, usize count, f64& log_scale) -> void
{
    std::vector<f64> scales{};
    scales.resize(count);
    download(scales.data(), state.device_scales, count);
    for (const auto scale : scales)
    {
        require(std::isfinite(scale), "zipup site scale {} is not finite", scale);
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
    const auto has_row = args.row_ket != nullptr and not args.row_ket->empty();
    require(has_row, "fused zipup row has no ket tensors");
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
    validate_descriptor(descriptor, layout);
    usize bytes{};
    require(
        elements_to_bytes(layout.output_elements, bytes),
        Status::bad_config,
        "zipup output of {} elements exceeds the i64 byte range",
        layout.output_elements
    );
    return static_cast<i64>(bytes);
}

auto execute(const QnpepsZipupMpoMpsDesc* descriptor, const QnpepsZipupMpoMpsArgs* args) -> void
{
    Layout layout{};
    validate_descriptor(descriptor, layout);
    validate_args(layout, args);

    auto session = make_session(static_cast<cudaStream_t>(args->stream));
    auto& linalg = session->linalg();
    const auto stream = linalg.stream();
    auto transient_arena = linalg.transient_arena();
    auto& carved = transient_arena.cursor();
    auto* fail_flag = carved.take<int>(2);
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
        .power_input_exponent = fail_flag + 1,
        .omegas = &omegas,
        .rangefinder_rng = &rangefinder_rng,
    };
    DEFER([&] { release_omegas(state); });

    constexpr cuFloatComplex one{1.0f, 0.0f};
    upload_async(linalg, initial_factor, &one, 1);
    zero_async(linalg, fail_flag, 1);

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

    check(cudaStreamSynchronize(stream));
    int fail_host{};
    download(&fail_host, fail_flag, 1);
    raise_rangefinder_failure(fail_host);
    accumulate_log_scales(state, layout.sites.size(), log_scale);
    *args->log_gauge = log_scale;
}
}
