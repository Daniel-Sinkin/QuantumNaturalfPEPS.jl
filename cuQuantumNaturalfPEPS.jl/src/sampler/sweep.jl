using CUDA

const _SWEEP_SEED_MULTIPLIER = UInt64(1000003)

Base.@kwdef struct SweepSiteArgs
    struct_size::UInt32
    row::Int32
    col::Int32
    has_below::Int32
    env_above_cur::Int32
    bond_above_l::Int32
    bond_above_r::Int32
    ket_bond_l::Int32
    ket_bond_r::Int32
    env_bond_l::Int32
    env_bond_r::Int32
    full_bond_l::Int32
    full_bond_r::Int32
end

Base.@kwdef struct _SweepSiteGeometry
    row::Int
    col::Int
    has_below::Bool
    env_above_cur::Int
    bond_above_l::Int
    bond_above_r::Int
    ket_bond_l::Int
    ket_bond_r::Int
    env_bond_l::Int
    env_bond_r::Int
    full_bond_l::Int
    full_bond_r::Int
end

struct SweepRow
    ket::Vector{SweepSiteArgs}
    env_unsampled::Vector{SweepSiteArgs}
    draw::Vector{SweepSiteArgs}
    env_above::Vector{SweepSiteArgs}
end

struct SweepPlan
    rows::Vector{SweepRow}
    site_entries::Int
end

Base.@kwdef mutable struct SweepHost{S}
    context::Ptr{Cvoid}
    seed_base::UInt64
    plan::SweepPlan
    seed::Vector{UInt64}
    seed_pointer::UInt
    samples_pointer::UInt
    logpc_pointer::UInt
    lognorm_pointer::UInt
    graph::Union{Nothing,CUDA.CuGraphExec}
    stream::S
    capture_enabled::Bool
    force_capture_failure::Bool
    warmed::Bool
    capture_attempted::Bool
    capture_state::Symbol
    capture_reason::Symbol
    open::Bool
end

@inline function _validate_sweep_selector(selector)::Symbol
    value = Symbol(selector)
    value === :compiled ||
        value === :julia ||
        throw(ArgumentError("sweep must be :compiled or :julia"))
    return value
end

@inline function _validate_sweep_capture(capture)::Symbol
    value = Symbol(capture)
    value === :auto ||
        value === :eager ||
        value === :fail ||
        throw(ArgumentError("sweep_capture must be :auto, :eager, or :fail"))
    return value
end

@inline _sweep_bond_dim(extent::Int, index::Int, dim_bond::Int)::Int =
    0 < index < extent ? dim_bond : 1

function _sweep_site_args(geometry::_SweepSiteGeometry)::SweepSiteArgs
    return SweepSiteArgs(;
        struct_size=UInt32(sizeof(SweepSiteArgs)),
        row=Int32(geometry.row),
        col=Int32(geometry.col),
        has_below=Int32(geometry.has_below),
        env_above_cur=Int32(geometry.env_above_cur),
        bond_above_l=Int32(geometry.bond_above_l),
        bond_above_r=Int32(geometry.bond_above_r),
        ket_bond_l=Int32(geometry.ket_bond_l),
        ket_bond_r=Int32(geometry.ket_bond_r),
        env_bond_l=Int32(geometry.env_bond_l),
        env_bond_r=Int32(geometry.env_bond_r),
        full_bond_l=Int32(geometry.full_bond_l),
        full_bond_r=Int32(geometry.full_bond_r),
    )
end

function _sweep_ket_bonds(config::QnpepsConfig, bond_above::Vector{Int}, row::Int)::Vector{Int}
    ly = Int(config.ly)
    dim_bond = Int(config.dim_bond)
    bonds = ones(Int, ly + 1)
    if row == 0
        for col in 0:ly
            bonds[col+1] = _sweep_bond_dim(ly, col, dim_bond)
        end
        return bonds
    end
    carried = 1
    for col in 0:(ly-1)
        bond_down = _sweep_bond_dim(Int(config.lx), row + 1, dim_bond)
        bond_right = _sweep_bond_dim(ly, col + 1, dim_bond)
        reduce_rows = carried * Int(config.dim_phys) * bond_down
        reduce_cols = bond_above[col+2] * bond_right
        carried = max(1, min(Int(config.chi_s), reduce_rows, reduce_cols))
        bonds[col+2] = carried
    end
    bonds[end] = 1
    return bonds
end

function _sweep_env_bonds(host, row::Int)::Vector{Int}
    ly = Int(host.config.ly)
    bonds = ones(Int, ly + 1)
    row + 1 == Int(host.config.lx) && return bonds
    for col in 0:(ly-1)
        base = 4 * (row * ly + col)
        bonds[col+1] = Int(host.dims[base+1])
        bonds[col+2] = Int(host.dims[base+4])
    end
    return bonds
end

function _sweep_full_bonds(config::QnpepsConfig, bond_above::Vector{Int}, row::Int)::Vector{Int}
    ly = Int(config.ly)
    dim_bond = Int(config.dim_bond)
    bonds = ones(Int, ly + 1)
    carried = 1
    for col in 0:(ly-1)
        bond_down = _sweep_bond_dim(Int(config.lx), row + 1, dim_bond)
        bond_right = _sweep_bond_dim(ly, col + 1, dim_bond)
        reduce_rows = carried * bond_down
        reduce_cols = bond_above[col+2] * bond_right
        carried = max(1, min(Int(config.chi_c), reduce_rows, reduce_cols))
        bonds[col+2] = carried
    end
    bonds[end] = 1
    return bonds
end

function SweepPlan(host)::SweepPlan
    config = host.config
    lx = Int(config.lx)
    ly = Int(config.ly)
    fast_mode = Int(config.sampling_mode) == 0
    rows = Vector{SweepRow}(undef, lx)
    bond_above = ones(Int, ly + 1)
    env_above_cur = 0
    site_entries = 0
    for row in 0:(lx-1)
        has_below = row + 1 < lx
        ket_bonds = _sweep_ket_bonds(config, bond_above, row)
        env_bonds = _sweep_env_bonds(host, row)
        full_bonds = fast_mode || row == 0 ? ket_bonds : _sweep_full_bonds(config, bond_above, row)
        ket = row == 0 ? SweepSiteArgs[] : Vector{SweepSiteArgs}(undef, ly)
        if row > 0
            for col in 0:(ly-1)
                geometry = _SweepSiteGeometry(;
                    row,
                    col,
                    has_below,
                    env_above_cur,
                    bond_above_l=bond_above[col+1],
                    bond_above_r=bond_above[col+2],
                    ket_bond_l=ket_bonds[col+1],
                    ket_bond_r=ket_bonds[col+2],
                    env_bond_l=env_bonds[col+1],
                    env_bond_r=env_bonds[col+2],
                    full_bond_l=full_bonds[col+1],
                    full_bond_r=full_bonds[col+2],
                )
                ket[col+1] = _sweep_site_args(geometry)
            end
        end
        env_unsampled = Vector{SweepSiteArgs}(undef, ly - 1)
        for (index, col) in enumerate((ly-1):-1:1)
            geometry = _SweepSiteGeometry(;
                row,
                col,
                has_below,
                env_above_cur,
                bond_above_l=bond_above[col+1],
                bond_above_r=bond_above[col+2],
                ket_bond_l=ket_bonds[col+1],
                ket_bond_r=ket_bonds[col+2],
                env_bond_l=env_bonds[col+1],
                env_bond_r=env_bonds[col+2],
                full_bond_l=full_bonds[col+1],
                full_bond_r=full_bonds[col+2],
            )
            env_unsampled[index] = _sweep_site_args(geometry)
        end
        draw = Vector{SweepSiteArgs}(undef, ly)
        for col in 0:(ly-1)
            geometry = _SweepSiteGeometry(;
                row,
                col,
                has_below,
                env_above_cur,
                bond_above_l=bond_above[col+1],
                bond_above_r=bond_above[col+2],
                ket_bond_l=ket_bonds[col+1],
                ket_bond_r=ket_bonds[col+2],
                env_bond_l=env_bonds[col+1],
                env_bond_r=env_bonds[col+2],
                full_bond_l=full_bonds[col+1],
                full_bond_r=full_bonds[col+2],
            )
            draw[col+1] = _sweep_site_args(geometry)
        end
        env_above = has_below ? Vector{SweepSiteArgs}(undef, ly) : SweepSiteArgs[]
        if has_below
            for col in 0:(ly-1)
                geometry = _SweepSiteGeometry(;
                    row,
                    col,
                    has_below,
                    env_above_cur,
                    bond_above_l=bond_above[col+1],
                    bond_above_r=bond_above[col+2],
                    ket_bond_l=ket_bonds[col+1],
                    ket_bond_r=ket_bonds[col+2],
                    env_bond_l=env_bonds[col+1],
                    env_bond_r=env_bonds[col+2],
                    full_bond_l=full_bonds[col+1],
                    full_bond_r=full_bonds[col+2],
                )
                env_above[col+1] = _sweep_site_args(geometry)
            end
            bond_above = copy(full_bonds)
            env_above_cur = 1 - env_above_cur
        end
        rows[row+1] = SweepRow(ket, env_unsampled, draw, env_above)
        site_entries += length(ket) + length(env_unsampled) + length(draw) + length(env_above)
    end
    return SweepPlan(rows, site_entries)
end

function _sweep_graph_policy(context::Ptr{Cvoid}, policy::Int32)::Tuple{Int32,Int32}
    state = Vector{Int32}(undef, 2)
    status = GC.@preserve state FFI.sweep_graph_policy(
        context,
        policy,
        pointer(state, 1),
        pointer(state, 2),
    )
    _check(; status, what="qnpeps_sweep_graph_policy")
    return state[1], state[2]
end

function SweepHost(host; capture=:auto)::SweepHost
    host.initialized || throw(ArgumentError("sampler context must be initialized"))
    capture_policy = _validate_sweep_capture(capture)
    capture_enabled = capture_policy !== :eager
    if capture_policy === :auto
        use_graph, _ = _sweep_graph_policy(host.handle, Int32(-1))
        capture_enabled = use_graph == 1
    end
    force_capture_failure = capture_policy === :fail
    seed = Vector{UInt64}(undef, 1)
    CUDA.pin(seed)
    seed_pointer = UInt(pointer(seed))
    args = host.batch_args[1]
    state = capture_enabled ? :cold : :eager
    reason =
        capture_enabled ? :not_attempted :
        capture_policy === :eager ? :explicit_eager : :route_not_capturable
    driver = SweepHost(;
        context=host.handle,
        seed_base=(host.config.seed * _SWEEP_SEED_MULTIPLIER),
        plan=SweepPlan(host),
        seed,
        seed_pointer,
        samples_pointer=args.samples_out,
        logpc_pointer=args.log_prob_config,
        lognorm_pointer=args.log_gauge,
        graph=nothing,
        stream=host.stream,
        capture_enabled,
        force_capture_failure,
        warmed=false,
        capture_attempted=false,
        capture_state=state,
        capture_reason=reason,
        open=true,
    )
    finalizer(close, driver)
    return driver
end

Base.isopen(driver::SweepHost)::Bool = driver.open

function Base.close(driver::SweepHost)::Nothing
    driver.open || return nothing
    CUDA.synchronize(driver.stream)
    executable = driver.graph
    if executable !== nothing
        graph = executable.graph
        finalize(executable)
        finalize(graph)
        driver.graph = nothing
    end
    driver.open = false
    return nothing
end

Base.copy(::SweepHost) = throw(ArgumentError("SweepHost cannot be copied"))

sweep_capture_mode(driver::SweepHost)::Symbol = driver.capture_state
sweep_capture_reason(driver::SweepHost)::Symbol = driver.capture_reason

@inline function _sweep_check(status::Cint, what::String)::Nothing
    _check(; status, what)
    return nothing
end

@inline function _sweep_build_ket_site!(
    context::Ptr{Cvoid},
    args::Vector{SweepSiteArgs},
    index::Int,
)::Nothing
    status = GC.@preserve args FFI.sweep_build_ket_site(context, pointer(args, index))
    _sweep_check(status, "qnpeps_sweep_build_ket_site")
    return nothing
end

@inline function _sweep_build_env_unsampled_site!(
    context::Ptr{Cvoid},
    args::Vector{SweepSiteArgs},
    index::Int,
)::Nothing
    status = GC.@preserve args FFI.sweep_build_env_unsampled_site(context, pointer(args, index))
    _sweep_check(status, "qnpeps_sweep_build_env_unsampled_site")
    return nothing
end

@inline function _sweep_draw_sigma_site!(
    context::Ptr{Cvoid},
    args::Vector{SweepSiteArgs},
    index::Int,
)::Nothing
    status = GC.@preserve args FFI.sweep_draw_sigma_site(context, pointer(args, index))
    _sweep_check(status, "qnpeps_sweep_draw_sigma_site")
    return nothing
end

@inline function _sweep_build_env_above_site!(
    context::Ptr{Cvoid},
    args::Vector{SweepSiteArgs},
    index::Int,
)::Nothing
    status = GC.@preserve args FFI.sweep_build_env_above_site(context, pointer(args, index))
    _sweep_check(status, "qnpeps_sweep_build_env_above_site")
    return nothing
end

function _enqueue_sweep!(driver::SweepHost)::Nothing
    context = driver.context
    status = FFI.sweep_begin(context, Ptr{UInt64}(driver.seed_pointer))
    _sweep_check(status, "qnpeps_sweep_begin")
    plan = driver.plan
    GC.@preserve plan begin
        for row in plan.rows
            for index in eachindex(row.ket)
                _sweep_build_ket_site!(context, row.ket, index)
            end
            for index in eachindex(row.env_unsampled)
                _sweep_build_env_unsampled_site!(context, row.env_unsampled, index)
            end
            for index in eachindex(row.draw)
                _sweep_draw_sigma_site!(context, row.draw, index)
            end
            for index in eachindex(row.env_above)
                _sweep_build_env_above_site!(context, row.env_above, index)
            end
        end
    end
    return nothing
end

function _finish_sweep!(driver::SweepHost)::Nothing
    status = FFI.sweep_finish(
        driver.context,
        Ptr{UInt8}(driver.samples_pointer),
        Ptr{Float64}(driver.logpc_pointer),
        Ptr{Float64}(driver.lognorm_pointer),
    )
    _sweep_check(status, "qnpeps_sweep_finish")
    return nothing
end

function _launch_sweep_graph!(driver::SweepHost)::Nothing
    executable = driver.graph
    executable === nothing && throw(ArgumentError("sweep graph is not instantiated"))
    _launch_dlenv_graph!(executable.handle, driver.stream.handle)
    return nothing
end

@inline function _prepare_sweep_seed!(
    driver::SweepHost,
    batch_base::UInt64,
    batch_id::UInt64,
)::Nothing
    driver.seed[1] = driver.seed_base + batch_base + batch_id
    return nothing
end

function _run_sweep!(driver::SweepHost, batch_base::UInt64, batch_id::UInt64)::Nothing
    driver.open || throw(ArgumentError("sweep driver is closed"))
    _prepare_sweep_seed!(driver, batch_base, batch_id)
    if driver.graph !== nothing
        _launch_sweep_graph!(driver)
        _finish_sweep!(driver)
        return nothing
    end
    if !driver.capture_enabled
        _enqueue_sweep!(driver)
        _finish_sweep!(driver)
        driver.warmed = true
        return nothing
    end
    if !driver.warmed
        _enqueue_sweep!(driver)
        _finish_sweep!(driver)
        driver.warmed = true
        driver.capture_state = :warmed
        driver.capture_reason = :warmup_complete
        return nothing
    end
    driver.capture_attempted = true
    graph = try
        CUDA.capture(; flags=CUDA.STREAM_CAPTURE_MODE_THREAD_LOCAL) do
            driver.force_capture_failure && error("forced capture failure")
            _enqueue_sweep!(driver)
        end
    catch
        driver.capture_enabled = false
        driver.capture_state = :fallback_eager
        driver.capture_reason =
            driver.force_capture_failure ? :forced_capture_failure : :capture_api_error
        _enqueue_sweep!(driver)
        _finish_sweep!(driver)
        return nothing
    end
    executable = try
        CUDA.instantiate(graph)
    catch
        finalize(graph)
        driver.capture_enabled = false
        driver.capture_state = :fallback_eager
        driver.capture_reason = :graph_instantiate_error
        _enqueue_sweep!(driver)
        _finish_sweep!(driver)
        return nothing
    end
    driver.graph = executable
    driver.capture_state = :captured
    driver.capture_reason = :capture_succeeded
    _launch_sweep_graph!(driver)
    _finish_sweep!(driver)
    return nothing
end
