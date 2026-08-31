#ifndef QNPEPS_TEST_SAMPLE_BATCH_H
#define QNPEPS_TEST_SAMPLE_BATCH_H

#include <cstdint>

inline constexpr uint64_t k_test_max_batch_size{2048};

[[nodiscard]] inline auto test_sample_batch_size(uint64_t count) -> uint64_t
{
    return count < k_test_max_batch_size ? count : k_test_max_batch_size;
}

#endif
