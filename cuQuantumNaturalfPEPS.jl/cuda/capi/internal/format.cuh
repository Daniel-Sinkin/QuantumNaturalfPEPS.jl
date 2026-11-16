#ifndef QNPEPS_CAPI_FORMAT_CUH
#define QNPEPS_CAPI_FORMAT_CUH

#include "capi/qnpeps.h"
#include "core/types.cuh"
#include "e2e/dans_qnpeps_e2e.h"
#include "eo/dans_qnpeps_eloc.h"

#include <format>

template <>
struct std::formatter<QnpepsConfig> : qnpeps::PlainFormatter
{
    auto format(const QnpepsConfig& value, std::format_context& context) const
        -> std::format_context::iterator
    {
        return std::format_to(
            context.out(),
            "lattice={}x{} dim_phys={} dim_bond={} chi_s={} chi_dl={} sampling_mode={} chi_c={} "
            "dlenv_route={} sampler_route={} cutoffs={},{},{}",
            value.lx,
            value.ly,
            value.dim_phys,
            value.dim_bond,
            value.chi_s,
            value.chi_dl,
            value.sampling_mode,
            value.chi_c,
            value.dlenv_truncation_route,
            value.sampler_truncation_route,
            value.dlenv_density_cutoff,
            value.sampler_density_cutoff,
            value.projected_density_cutoff
        );
    }
};

template <>
struct std::formatter<QnpepsE2eConfig> : qnpeps::PlainFormatter
{
    auto format(const QnpepsE2eConfig& value, std::format_context& context) const
        -> std::format_context::iterator
    {
        return std::format_to(
            context.out(),
            "lattice={}x{} dim_phys={} dim_bond={} chi_s={} chi_dl={} chi_eo={} meo={} "
            "sampling_mode={} contract_dim={} sample_batch={} route={} cutoffs={},{},{},{},{}",
            value.lx,
            value.ly,
            value.dim_phys,
            value.dim_bond,
            value.chi_s,
            value.chi_dl,
            value.chi_eo,
            value.meo,
            value.sampling_mode,
            value.contract_dim,
            value.sample_batch,
            value.truncation_route,
            value.dlenv_density_cutoff,
            value.sampler_density_cutoff,
            value.projected_density_cutoff,
            value.local_energy_density_cutoff,
            value.local_energy_svd_cutoff
        );
    }
};

template <>
struct std::formatter<QnpepsElocConfig> : qnpeps::PlainFormatter
{
    auto format(const QnpepsElocConfig& value, std::format_context& context) const
        -> std::format_context::iterator
    {
        return std::format_to(
            context.out(),
            "lattice={}x{} dim_phys={} dim_bond={} chi_eo={} meo={} route={} density_cutoff={} "
            "svd_cutoff={}",
            value.lx,
            value.ly,
            value.dim_phys,
            value.dim_bond,
            value.chi_eo,
            value.meo,
            value.truncation_route,
            value.density_cutoff,
            value.svd_cutoff
        );
    }
};

template <>
struct std::formatter<QnpepsGramDesc> : qnpeps::PlainFormatter
{
    auto format(const QnpepsGramDesc& value, std::format_context& context) const
        -> std::format_context::iterator
    {
        return std::format_to(
            context.out(),
            "lattice={}x{} dim_phys={} dim_bond={} consumer={} reserved={} n_samples={}",
            value.lx,
            value.ly,
            value.dim_phys,
            value.dim_bond,
            value.consumer,
            value.reserved,
            value.n_samples
        );
    }
};

template <>
struct std::formatter<QnpepsMinsrDesc> : qnpeps::PlainFormatter
{
    auto format(const QnpepsMinsrDesc& value, std::format_context& context) const
        -> std::format_context::iterator
    {
        return std::format_to(
            context.out(),
            "lattice={}x{} dim_phys={} dim_bond={} diagnostics={} reserved={} n_samples={} "
            "host_tile_bytes={}",
            value.lx,
            value.ly,
            value.dim_phys,
            value.dim_bond,
            value.diagnostics,
            value.reserved,
            value.n_samples,
            value.host_tile_bytes
        );
    }
};

template <>
struct std::formatter<QnpepsZipupMpoMpsDesc> : qnpeps::PlainFormatter
{
    auto format(const QnpepsZipupMpoMpsDesc& value, std::format_context& context) const
        -> std::format_context::iterator
    {
        return std::format_to(
            context.out(),
            "num_sites={} maxdim={} reserved={}",
            value.num_sites,
            value.maxdim,
            value.reserved
        );
    }
};

template <>
struct std::formatter<QnpepsDensityWorkspaceQuery> : qnpeps::PlainFormatter
{
    auto format(const QnpepsDensityWorkspaceQuery& value, std::format_context& context) const
        -> std::format_context::iterator
    {
        return std::format_to(
            context.out(),
            "num_sites={} input_bond={} operator_bond={} output_dimension={} upper_bond={} "
            "lanes={}",
            value.num_sites,
            value.input_bond,
            value.operator_bond,
            value.output_dimension,
            value.upper_bond,
            value.lanes
        );
    }
};

#endif
