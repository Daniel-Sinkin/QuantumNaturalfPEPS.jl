#pragma once

#include "../eo.cuh"

#include <algorithm>
#include <cstdint>

#line 16 "cuda/linalg/experimental_svd.cuh"

namespace qn_eloc::e0191
{
using namespace qnpeps;

struct GesvdaWorkspace
{
    cf* sketch{};
    cf* projection{};
    cf* left{};
    cf* right{};
    cf* q_candidate{};
    cf* r_candidate{};
    cf* direct_input{};
    f32* singular{};
    f64* total_weight{};
    f64* component_weight{};
    int* effective_rank{};
    int* info{};
    char* qr_scratch{};
    cf* solver_scratch{};
    usize solver_scratch_count{};
};

using WorkspaceCursor = ArenaCursor;

inline auto condition_spectrum_rank(int available, int cap) -> int
{
    static_cast<void>(available);
    return cap;
}

}
