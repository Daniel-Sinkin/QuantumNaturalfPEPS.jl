#ifndef QNPEPS_ELOC_COMMON_CUH
#define QNPEPS_ELOC_COMMON_CUH

#include "core/error.cuh"
#include "core/predicates.cuh"
#include "core/types.cuh"
#include "dans_qnpeps_eloc.h"
#include "eloc_fixed.cuh"

#include <cuda_runtime.h>

using qnpeps::f32;
using qnpeps::f64;
using qnpeps::i32;
using qnpeps::i64;
using qnpeps::u16;
using qnpeps::u32;
using qnpeps::u64;
using qnpeps::u8;
using qnpeps::usize;

inline constexpr usize CUDA_MALLOC_ALIGN{qnpeps::k_device_malloc_align};
inline constexpr u64 k_max_batch_size{qnpeps::k_max_batch_size};

using cf = qn_eloc::fx::cf;

namespace qn
{
static_assert(QNPEPS_ELOC_OK == static_cast<int>(QNPEPS_OK));
static_assert(QNPEPS_ELOC_ERR_NULL_ARG == static_cast<int>(QNPEPS_ERR_NULL_ARG));
static_assert(QNPEPS_ELOC_ERR_BAD_CONFIG == static_cast<int>(QNPEPS_ERR_BAD_CONFIG));
static_assert(QNPEPS_ELOC_ERR_BAD_VERSION == static_cast<int>(QNPEPS_ERR_BAD_VERSION));
static_assert(QNPEPS_ELOC_ERR_CUDA == static_cast<int>(QNPEPS_ERR_CUDA));
static_assert(QNPEPS_ELOC_ERR_OOM == static_cast<int>(QNPEPS_ERR_OOM));
static_assert(QNPEPS_ELOC_ERR_INTERNAL == static_cast<int>(QNPEPS_ERR_INTERNAL));

inline auto root_status(qnpeps_eloc_status status) -> qnpeps_status
{
    return static_cast<qnpeps_status>(status);
}

inline auto err_state() -> qnpeps_eloc_status
{
    return static_cast<qnpeps_eloc_status>(qnpeps::err_state());
}
inline void clear_err()
{
    qnpeps::reset_err();
}
inline void set_err(qnpeps_eloc_status s)
{
    qnpeps::set_err(root_status(s));
}
inline void set_backend_err(
    qnpeps_eloc_status status, const char* backend, int code, const char* file, int line
)
{
    qnpeps::set_backend_err_at(root_status(status), backend, code, file, line);
}
}

#endif
