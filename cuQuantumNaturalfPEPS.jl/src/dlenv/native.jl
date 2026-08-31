
function _double_layer_native(
    device_peps::CuPeps;
    chi_s::Integer=device_peps.dim_bond,
    chi_dl::Integer=device_peps.dim_bond,
    route=:default,
    density_cutoff::Real=1.0e-13,
)::CuDlenv
    config = _cfg_of(
        device_peps;
        chi_s=chi_s,
        chi_dl=chi_dl,
        dlenv_truncation_route=route,
        dlenv_density_cutoff=density_cutoff,
    )

    n_bytes = _dlenv_bytes(; config)
    n_bytes >= 0 || throw(
        ArgumentError(
            "invalid double-layer configuration: $device_peps, chi_s=$chi_s, chi_dl=$chi_dl",
        ),
    )
    data = CUDA.zeros(UInt8, n_bytes)
    logs = CUDA.zeros(Float64, device_peps.lx - 1)

    GC.@preserve device_peps data logs begin
        _ffi_build_dlenv(;
            config,
            peps=pointer(device_peps.data),
            dlenv=pointer(data),
            cumulative_row_logs=pointer(logs),
        )
    end

    return CuDlenv(
        data,
        Array(logs),
        device_peps.lx,
        device_peps.ly,
        device_peps.dim_phys,
        device_peps.dim_bond,
        chi_s,
        chi_dl,
    )
end

function double_layer(
    device_peps::CuPeps;
    chi_s::Integer=device_peps.dim_bond,
    chi_dl::Integer=device_peps.dim_bond,
    route=:default,
    density_cutoff::Real=1.0e-13,
    workspace::Union{Nothing,ZipupWorkspace}=nothing,
)::CuDlenv
    return double_layer_rowwise(device_peps; chi_s, chi_dl, route, density_cutoff, workspace)
end
