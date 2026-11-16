#ifndef QNPEPS_ELOC_COMMON_CUH
#define QNPEPS_ELOC_COMMON_CUH

#include "core/error.hpp"
#include "core/predicates.cuh"
#include "core/types.cuh"
#include "dans_qnpeps_eloc.h"
#include "eloc_fixed.cuh"
#include "routes.cuh"

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

using qnpeps::check;
using qnpeps::fail;
using qnpeps::guarded;
using qnpeps::guarded_keeping_message;
using qnpeps::guarded_labelled;
using qnpeps::require;
using qnpeps::require_struct_size;
using qnpeps::Status;
using qnpeps::unchecked;

inline constexpr usize CUDA_MALLOC_ALIGN{qnpeps::k_device_malloc_align};
inline constexpr u64 k_max_batch_size{qnpeps::k_max_batch_size};

using cf = qn_eloc::fx::cf;

#endif
