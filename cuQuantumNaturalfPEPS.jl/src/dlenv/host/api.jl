
function dlenv_capture_mode(host::DlenvHost)::Symbol
    return host.capture_state
end

function dlenv_capture_reason(host::DlenvHost)::Symbol
    return host.capture_reason
end

function _build_dlenv!(host::DlenvHost, device_peps::CuPeps, requested_generation::Int)::CuDlenv
    _validate_dlenv_host(host, device_peps)
    if requested_generation == host.generation
        host.active_lane != 0 || throw(ArgumentError("dl-env generation is not built"))
        return host.lanes[host.active_lane]
    end
    requested_generation > host.generation ||
        throw(ArgumentError("dl-env generation must not move backward"))
    lane = host.build_count % _DLENV_HOST_LANES + 1
    density_route = host.config.dlenv_truncation_route == TRUNCATION_DENSITY
    if density_route
        _warm_dlenv_rows!(host, device_peps)
        host.args_ready = false
        _prepare_dlenv_layout!(host, device_peps)
    elseif host.layout_ready
        _enqueue_prepared_rows!(host, device_peps)
    else
        _warm_dlenv_rows!(host, device_peps)
        _prepare_dlenv_layout!(host, device_peps)
    end
    _extract_dlenv_logs!(host, lane)
    _pack_dlenv_with_policy!(host, lane)
    host.generation = requested_generation
    host.active_lane = lane
    host.build_count += 1
    return host.lanes[lane]
end

function build_dlenv!(host::DlenvHost, device_peps::CuPeps, generation::Int)::CuDlenv
    return _build_dlenv!(host, device_peps, generation)
end

function build_dlenv!(host::DlenvHost, device_peps::CuPeps, generation::Integer)::CuDlenv
    return _build_dlenv!(host, device_peps, Int(generation))
end

function build_dlenv!(
    host::DlenvHost,
    device_peps::CuPeps;
    generation::Integer=(host.generation + 1),
)::CuDlenv
    return _build_dlenv!(host, device_peps, Int(generation))
end
