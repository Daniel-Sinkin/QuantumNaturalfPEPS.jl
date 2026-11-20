#include "core/error.hpp"
#include "linalg/transfer.cuh"
#include "tensor/permutation.cuh"

#include <cstdio>
#include <cstring>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
using qnpeps::i64;
using qnpeps::usize;
using qnpeps::operator""_uz;

constexpr i64 k_destination_padding{3};
constexpr i64 k_source_padding{5};
constexpr float k_padding_value{-7.5f};
constexpr float k_lane_real_step{0.25f};
constexpr float k_lane_imaginary_step{0.5f};

int failures{};

auto expect(bool condition, std::string_view name, const std::string& detail = {}) -> void
{
    std::printf(
        "[permutation_test] %s %.*s",
        condition ? "ok" : "FAIL",
        static_cast<int>(name.size()),
        name.data()
    );
    if (not condition and not detail.empty()) std::printf(" (%s)", detail.c_str());
    std::printf("\n");
    if (not condition) ++failures;
}

class DeviceAllocation
{
  public:
    explicit DeviceAllocation(usize count)
    {
        qnpeps::check(cudaMalloc(&pointer_, count * sizeof(cuFloatComplex)));
    }
    ~DeviceAllocation() { qnpeps::maybe_free_device(pointer_); }
    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation(DeviceAllocation&&) = delete;
    auto operator=(const DeviceAllocation&) -> DeviceAllocation& = delete;
    auto operator=(DeviceAllocation&&) -> DeviceAllocation& = delete;

    [[nodiscard]] auto get() const noexcept -> cuFloatComplex* { return pointer_; }

  private:
    cuFloatComplex* pointer_{};
};

struct PermutationCase
{
    std::vector<int> dims_in{};
    std::vector<int> perm{};
    int batch_count{};
    bool conjugate{};
    bool broadcast_source{};
};

[[nodiscard]] auto source_stride(const PermutationCase& sample) -> i64
{
    if (sample.broadcast_source) return 0;
    return static_cast<i64>(qnpeps::Shape{sample.dims_in}.num_elems()) + k_source_padding;
}

[[nodiscard]] auto destination_stride(const PermutationCase& sample) -> i64
{
    return static_cast<i64>(qnpeps::Shape{sample.dims_in}.num_elems()) + k_destination_padding;
}

[[nodiscard]] auto source_values(const PermutationCase& sample) -> std::vector<cuFloatComplex>
{
    const auto element_count = qnpeps::Shape{sample.dims_in}.num_elems();
    const auto lane_count = sample.broadcast_source ? 1_uz : static_cast<usize>(sample.batch_count);
    const auto stride = element_count + static_cast<usize>(k_source_padding);
    std::vector<cuFloatComplex> values(stride * lane_count, cuFloatComplex{k_padding_value, 0.0f});
    for (auto lane = 0_uz; lane < lane_count; ++lane)
    {
        const auto lane_value = static_cast<float>(lane);
        for (auto element = 0_uz; element < element_count; ++element)
        {
            const auto element_value = static_cast<float>(element);
            values[lane * stride + element] = cuFloatComplex{
                element_value + 1.0f + k_lane_real_step * lane_value,
                -2.0f * element_value - 1.0f - k_lane_imaginary_step * lane_value
            };
        }
    }
    return values;
}

[[nodiscard]] auto expected_values(
    const PermutationCase& sample, const std::vector<cuFloatComplex>& source
) -> std::vector<cuFloatComplex>
{
    const auto rank = sample.dims_in.size();
    const auto element_count = qnpeps::Shape{sample.dims_in}.num_elems();
    const auto lane_count = static_cast<usize>(sample.batch_count);
    const auto output_stride = static_cast<usize>(destination_stride(sample));
    const auto input_stride = static_cast<usize>(source_stride(sample));
    std::vector<usize> input_strides(rank, 1_uz);
    for (auto axis = 1_uz; axis < rank; ++axis)
    {
        const auto previous_extent = static_cast<usize>(sample.dims_in[axis - 1]);
        input_strides[axis] = input_strides[axis - 1] * previous_extent;
    }
    std::vector<cuFloatComplex> expected(
        output_stride * lane_count, cuFloatComplex{k_padding_value, k_padding_value}
    );
    for (auto output_element = 0_uz; output_element < element_count; ++output_element)
    {
        auto remainder = output_element;
        auto input_element = 0_uz;
        for (auto axis = 0_uz; axis < rank; ++axis)
        {
            const auto input_axis = static_cast<usize>(sample.perm[axis]);
            const auto extent = static_cast<usize>(sample.dims_in[input_axis]);
            input_element += (remainder % extent) * input_strides[input_axis];
            remainder /= extent;
        }
        for (auto lane = 0_uz; lane < lane_count; ++lane)
        {
            const auto value = source[lane * input_stride + input_element];
            expected[lane * output_stride + output_element] =
                sample.conjugate ? cuConjf(value) : value;
        }
    }
    return expected;
}

struct PermutationRun
{
    std::vector<cuFloatComplex> output{};
    std::vector<cuFloatComplex> expected{};
};

template <typename Permute>
[[nodiscard]] auto run_case(const PermutationCase& sample, Permute&& permute) -> PermutationRun
{
    const auto source = source_values(sample);
    const DeviceAllocation device_source{source.size()};
    qnpeps::upload(device_source.get(), source.data(), source.size());

    const auto expected = expected_values(sample, source);
    const std::vector<cuFloatComplex> padding(
        expected.size(), cuFloatComplex{k_padding_value, k_padding_value}
    );
    const DeviceAllocation device_destination{expected.size()};
    qnpeps::upload(device_destination.get(), padding.data(), padding.size());

    const qnpeps::PermuteOp op{
        .dst = {device_destination.get(), destination_stride(sample)},
        .src = {device_source.get(), source_stride(sample)},
        .dims_in = qnpeps::Shape{sample.dims_in},
        .perm = qnpeps::Permutation{sample.perm},
        .batch_count = sample.batch_count,
        .conjugate = sample.conjugate,
    };
    permute(op);
    qnpeps::check(cudaDeviceSynchronize());

    std::vector<cuFloatComplex> output(expected.size());
    qnpeps::download(output.data(), device_destination.get(), output.size());
    return {.output = std::move(output), .expected = expected};
}

[[nodiscard]] auto is_bitwise_equal(
    const std::vector<cuFloatComplex>& left, const std::vector<cuFloatComplex>& right
) -> bool
{
    const auto bytes = left.size() * sizeof(cuFloatComplex);
    return left.size() == right.size() and std::memcmp(left.data(), right.data(), bytes) == 0;
}

auto small_shape_paths() -> void
{
    const PermutationCase sample{
        .dims_in = {3, 5, 2, 7},
        .perm = {2, 0, 3, 1},
        .batch_count = 4,
        .conjugate = true,
    };
    qnpeps::PermutationCache cache{};
    const auto cached = run_case(
        sample, [&](const qnpeps::PermuteOp& op) { qnpeps::permute_batched(cache, op, nullptr); }
    );
    expect(cache.size() == 1, "a small permutation caches one index map");
    expect(
        is_bitwise_equal(cached.output, cached.expected),
        "the cached map path conjugates, transposes, and keeps the padding"
    );

    const auto strided = run_case(
        sample, [](const qnpeps::PermuteOp& op) { qnpeps::permute_batched_strided(op, nullptr); }
    );
    expect(
        is_bitwise_equal(strided.output, strided.expected),
        "the strided path conjugates, transposes, and keeps the padding"
    );
    expect(
        is_bitwise_equal(strided.output, cached.output),
        "the strided path matches the cached map path bit for bit"
    );
    cache.release();
}

auto large_shape_paths() -> void
{
    const std::vector<PermutationCase> samples{
        {.dims_in = {64, 129, 2, 65}, .perm = {3, 1, 0, 2}, .batch_count = 3, .conjugate = true},
        {.dims_in = {65, 2, 129, 64},
         .perm = {1, 3, 2, 0},
         .batch_count = 2,
         .broadcast_source = true},
    };
    for (const auto& sample : samples)
    {
        const auto element_count = qnpeps::Shape{sample.dims_in}.num_elems();
        expect(
            element_count > qnpeps::k_max_cached_index_map_elements,
            "the large shape exceeds the cached index map bound"
        );
        qnpeps::PermutationCache cache{};
        const auto run = run_case(
            sample,
            [&](const qnpeps::PermuteOp& op) { qnpeps::permute_batched(cache, op, nullptr); }
        );
        expect(cache.size() == 0, "a large permutation leaves the index map cache empty");
        expect(
            is_bitwise_equal(run.output, run.expected),
            "a large permutation through the cache entry point matches the host reference"
        );
    }
}
}

auto main() -> int
{
    const auto status = qnpeps::guarded(
        []
        {
            small_shape_paths();
            large_shape_paths();
        }
    );
    expect(status == QNPEPS_OK, "the permutation cases run", std::string{qnpeps_last_error()});
    std::printf("[permutation_test] failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
