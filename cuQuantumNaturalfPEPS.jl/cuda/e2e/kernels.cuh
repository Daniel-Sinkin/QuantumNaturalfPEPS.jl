#ifndef QNPEPS_E2E_KERNELS_CUH
#define QNPEPS_E2E_KERNELS_CUH

#include "update.cuh"

namespace qn_e2e
{

struct UpdateF32Args
{
    const UpdateSite* sites;
    cf* peps;
    const cf* theta;
    f32 rate;
};

struct UpdateF64Args
{
    const UpdateSite* sites;
    zd* state;
    const cf* theta;
    cf* peps;
    f64 rate;
};

__global__ auto cu_update_f32(UpdateF32Args args) -> void;
__global__ auto cu_update_f64(UpdateF64Args args) -> void;

}

#endif
