#ifndef QNPEPS_E2E_UPDATE_CUH
#define QNPEPS_E2E_UPDATE_CUH

#include "common.cuh"

#include <vector>

namespace qn_e2e
{

struct UpdateSite
{
    i64 offset{};
    i64 count{};
    i32 dw{};
    i32 ds{};
    i32 de{};
    i32 dn{};
    i32 dp{};
};

struct zd
{
    f64 re;
    f64 im;
};

static_assert(sizeof(zd) == 2 * sizeof(f64));

auto build_update_sites(const QnpepsE2eConfig& cfg) -> std::vector<UpdateSite>;

auto launch_update_f32(
    const UpdateSite* sites,
    int site_count,
    cf* peps_f32_io,
    const cf* theta_dot,
    f64 learning_rate,
    cudaStream_t stream
) -> qnpeps_e2e_status;

auto launch_update_f64(
    const UpdateSite* sites,
    int site_count,
    zd* state_f64_io,
    const cf* theta_dot,
    cf* peps_f32_out,
    f64 learning_rate,
    cudaStream_t stream
) -> qnpeps_e2e_status;

}

#endif
