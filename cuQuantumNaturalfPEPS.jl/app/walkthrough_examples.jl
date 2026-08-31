using CUDA
using LinearAlgebra
using Printf
using cuQuantumNaturalfPEPS

include(joinpath(@__DIR__, "common.jl"))

const OUT = normpath(
    joinpath(@__DIR__, "..", "..", "documentation", "dossier_examples", "out_cuda_heavy"),
)
mkpath(OUT)
const LOG = Ref{IO}(devnull)

function emit(io::IO, args...)
    println(io, args...)
    println(LOG[], args...)
    flush(io)
    flush(LOG[])
    return nothing
end

function emitf(io::IO, fmt, args...)
    emit(io, Printf.format(Printf.Format(fmt), args...))
    return nothing
end

function show_vector(io::IO, label, v)
    emit(io, label, " length=", length(v))
    for i in eachindex(v)
        x = v[i]
        if x isa Complex
            emitf(io, "  [%d] % .12e %+.12eim", i, real(x), imag(x))
        else
            emitf(io, "  [%d] % .16e", i, Float64(x))
        end
    end
    return nothing
end

function show_matrix(io::IO, label, A)
    emit(io, label, " size=", size(A))
    for i in axes(A, 1)
        emit(io, "  ", join([@sprintf("% .10f%+.10fim", real(A[i, j]), imag(A[i, j])) for j in axes(A, 2)], "  "))
    end
    return nothing
end

function section(body, name::String)
    path = joinpath(OUT, name * ".txt")
    open(path, "w") do io
        emit(io, "==== ", name, " ====")
        try
            body(io)
            emit(io, "[status] ok")
        catch err
            emit(io, "[status] failed ", sprint(showerror, err))
            for line in stacktrace(catch_backtrace())[1:min(end, 15)]
                emit(io, "  ", line)
            end
        end
    end
    return nothing
end

deterministic_unitary(n::Int) =
    Matrix(qr([cos((i * j) / 3) + im * sin((i + 2j) / 5) for i in 1:n, j in 1:n]).Q)

function fixed_panel_5x4()
    u = deterministic_unitary(5)
    v = deterministic_unitary(4)
    return ComplexF64.(u[:, 1:4] * Diagonal([1.0, 0.5, 0.05, 0.002]) * v')
end

bd(axis_len, pos, dim_bond) = (pos <= 0 || pos >= axis_len) ? 1 : dim_bond

proj_dims(lx, ly, dim_bond, i, c) = (
    bd(ly, c, dim_bond),
    bd(lx, i + 1, dim_bond),
    bd(ly, c + 1, dim_bond),
    bd(lx, i, dim_bond),
)

function site_tensors_2x2()
    lx = 2
    ly = 2
    dim_phys = 2
    dim_bond = 2
    sites = Dict{Tuple{Int,Int},Array{ComplexF64,5}}()
    for i in 0:(lx - 1), c in 0:(ly - 1)
        w, s, e, n = proj_dims(lx, ly, dim_bond, i, c)
        t = zeros(ComplexF64, w, s, e, n, dim_phys)
        idx = 0
        for p in 1:dim_phys, nn in 1:n, ee in 1:e, ss in 1:s, ww in 1:w
            idx += 1
            key = (i * ly + c) * 37 + idx
            t[ww, ss, ee, nn, p] =
                cospi(key / 17) / (1 + 0.25 * idx) + im * sinpi(key / 23) / (2 + 0.1 * idx)
        end
        sites[(i, c)] = t
    end
    return sites
end

function packed_peps_2x2(sites)
    buffer = ComplexF32[]
    for i in 0:1, c in 0:1
        append!(buffer, vec(ComplexF32.(sites[(i, c)])))
    end
    return buffer
end

const CPU_REFERENCE = Dict(
    "eo_logabs" => [-5.0836424514816692e-01, -1.4884727645898850e+00, -9.3249824822737215e-01],
    "eo_phase" => [7.4660080320880806e-01, -1.1012774435011006e+00, 2.2287432262776372e+00],
    "eo_eloc_real" => [4.0000000000000000e+00, -6.9231433541490235e+00, -3.7490157060739668e+00],
    "eo_eloc_imag" => [0.0, 1.2992967816239091e+01, -7.6337686794194575e+00],
    "minsr_ess" => 2.9843633277556627e+00,
    "minsr_e_mean_real" => -1.2158136056989679e+00,
    "minsr_e_mean_imag" => 4.9999999999999996e-02,
    "minsr_e_var" => 1.2178499883672378e-01,
    "minsr_theta_dot" => ComplexF64[
        -1.518920966300e-01 - 5.826879725660e-02im,
        -9.563014566181e-02 - 9.872339732298e-03im,
        4.845754873551e-02 - 1.832033787513e-01im,
        -1.497162313239e-01 + 9.344613486089e-02im,
        1.700921866377e-01 - 3.006502141247e-02im,
        0.0 + 0.0im,
        -1.705361161059e-02 - 1.430318346302e-01im,
        -1.786714290660e-01 + 1.824489022928e-02im,
    ],
    "zipup_log_gauge" => 2.9966588125134370e+00,
)

const OK_ROWS_CPU = ComplexF64[
    0.80+0.10im 0.20-0.30im 0.50+0.40im -0.10+0.20im
    0.30-0.20im 0.70+0.05im -0.40+0.15im 0.60-0.25im
    -0.50+0.35im 0.45+0.60im 0.25-0.10im 0.15+0.55im
]
const SPINS_CPU = UInt8[0 1 0 1; 1 1 0 1; 0 0 0 0]

function walkthrough_examples(arguments=ARGS)::Nothing
    options = parse_app_options("walkthrough_examples.jl", arguments)
    app_dry_run("walkthrough_examples.jl", options) && return nothing
    CUDA.functional() || error("walkthrough_examples.jl requires CUDA")

section("environment") do io
    emit(io, "[env] QNPEPS_LIB ", get(ENV, "QNPEPS_LIB", "unset"))
    emit(io, "[env] active project ", Base.active_project())
    emit(io, "[env] QNPEPS_SAMPLER_TRUNC ", get(ENV, "QNPEPS_SAMPLER_TRUNC", "unset"))
    emit(io, "[env] QNPEPS_DLENV_TRUNC ", get(ENV, "QNPEPS_DLENV_TRUNC", "unset"))
    emit(io, "[env] QNPEPS_E0191_RF_ROUTE ", get(ENV, "QNPEPS_E0191_RF_ROUTE", "unset"))
    emit(io, "[env] julia ", VERSION)
    emit(io, "[cuda] devices ", CUDA.ndevices())
    for d in CUDA.devices()
        emit(io, "[cuda] device ", d, " name ", CUDA.name(d), " capability ", CUDA.capability(d))
    end
    emit(io, "[cuda] runtime ", CUDA.runtime_version())
    emit(io, "[abi] core capi version ", unsafe_string(FFI.capi_version()))
    try
        emit(io, "[abi] e2e version ", e2e_version())
        emit(io, "[abi] eloc version ", eloc_version())
        emit(io, "[abi] sampler version ", sampler_version())
    catch err
        emit(io, "[abi] step layer versions unavailable ", sprint(showerror, err))
    end
end

section("truncation_panels") do io
    a = fixed_panel_5x4()
    exact = svd(a)
    show_matrix(io, "[panel] A", a)
    show_vector(io, "[panel] exact singular values", exact.S)
    emitf(io, "[panel] frobenius norm %.16e", norm(a))
    for rank in (2, 3)
        emit(io, "")
        emit(io, "---- device batched_rangefinder rank ", rank, " ----")
        input = CuArray(reshape(ComplexF32.(a), 5, 4, 1))
        qs, rs = batched_rangefinder(input, rank; seed = 1)
        q = Array(qs)[:, :, 1]
        r = Array(rs)[:, :, 1]
        show_matrix(io, "[device] Q", q)
        show_matrix(io, "[device] R", r)
        emitf(io, "[device] orthogonality residual %.6e", norm(q' * q - I))
        emitf(io, "[device] truncation error %.16e", norm(ComplexF32.(a) - q * r))
        emitf(io, "[device] relative truncation error %.16e", norm(ComplexF32.(a) - q * r) / norm(a))
        best = rank < length(exact.S) ? sqrt(sum(exact.S[(rank + 1):end] .^ 2)) : 0.0
        emitf(io, "[host] optimal rank-%d error %.16e", rank, best)
        emitf(io, "[host] device error over optimal %.6f", norm(ComplexF32.(a) - q * r) / best)
    end
end

section("zipup_three_site") do io
    d = 2
    chi = 2
    mpo = [ComplexF64[
        ((a - 1) * 8 + (p - 1) * 4 + (q - 1) * 2 + (b - 1) + 1) / 10 +
        im * ((a + p + q + b) % 3) / 7
        for a in 1:(s == 1 ? 1 : chi), p in 1:d, q in 1:d, b in 1:(s == 3 ? 1 : chi)
    ] for s in 1:3]
    mps = [ComplexF64[
        ((a - 1) * 4 + (q - 1) * 2 + (b - 1) + 1) / 8 - im * ((a + 2q + b) % 4) / 9
        for a in 1:(s == 1 ? 1 : chi), q in 1:d, b in 1:(s == 3 ? 1 : chi)
    ] for s in 1:3]
    for s in 1:3
        emit(io, "[input] mpo site ", s, " dims ", size(mpo[s]), " mps site ", s, " dims ", size(mps[s]))
    end

    exactamp = zeros(ComplexF64, d, d, d)
    for p1 in 1:d, p2 in 1:d, p3 in 1:d
        acc = zero(ComplexF64)
        for a1 in 1:chi, a2 in 1:chi, b1 in 1:chi, b2 in 1:chi, q1 in 1:d, q2 in 1:d, q3 in 1:d
            acc +=
                mpo[1][1, q1, p1, a1] * mpo[2][a1, q2, p2, a2] * mpo[3][a2, q3, p3, 1] *
                mps[1][1, q1, b1] * mps[2][b1, q2, b2] * mps[3][b2, q3, 1]
        end
        exactamp[p1, p2, p3] = acc
    end
    show_vector(io, "[host] exact amplitudes", vec(exactamp))

    result = zipup_mpo_mps(
        [CuArray(ComplexF32.(t)) for t in mpo],
        [CuArray(ComplexF32.(t)) for t in mps];
        maxdim = chi,
    )
    for s in 1:3
        emit(io, "[device] output site ", s, " dims ", size(result.mps[s]))
    end
    emitf(io, "[device] log_gauge %.16e", result.log_gauge)
    emitf(io, "[host] cpu reference log_gauge %.16e", CPU_REFERENCE["zipup_log_gauge"])
    emitf(io, "[host] log_gauge difference %.6e", abs(result.log_gauge - CPU_REFERENCE["zipup_log_gauge"]))

    o1 = Array(result.mps[1])
    o2 = Array(result.mps[2])
    o3 = Array(result.mps[3])
    recon = zeros(ComplexF64, d, d, d)
    for p1 in 1:d, p2 in 1:d, p3 in 1:d
        acc = zero(ComplexF64)
        for a in axes(o1, 3), b in axes(o2, 3)
            acc += o1[1, p1, a] * o2[a, p2, b] * o3[b, p3, 1]
        end
        recon[p1, p2, p3] = acc * exp(result.log_gauge)
    end
    show_vector(io, "[device] zipped amplitudes", vec(recon))
    emitf(io, "[device] reconstruction error %.16e", norm(recon - exactamp))
    emitf(io, "[device] relative reconstruction error %.16e", norm(recon - exactamp) / norm(exactamp))
end

section("eo_two_by_two") do io
    lx = 2
    ly = 2
    dim_phys = 2
    dim_bond = 2
    chi_eo = 2
    meo = 4
    sites = site_tensors_2x2()
    for i in 0:1, c in 0:1
        emit(io, "[geometry] site (", i, ",", c, ") projected dims (W,S,E,N) = ",
            proj_dims(lx, ly, dim_bond, i, c))
    end
    config = QnpepsElocConfig(; lx, ly, dim_bond, chi_eo, meo, dim_phys)
    compact = eloc_compact_count(config)
    emit(io, "[geometry] compact_count ", compact, " dense_count ", dim_phys * compact)

    configurations = UInt8[0 0 0 0; 0 1 1 0; 1 0 0 1]
    ns = size(configurations, 1)
    peps_host = packed_peps_2x2(sites)
    peps = CuArray(peps_host)
    samples = CuArray(vec(permutedims(configurations, (2, 1))))
    logpsi = CUDA.zeros(Float64, 2 * ns)
    e_loc = CUDA.zeros(Float64, 2 * ns)
    rows = CUDA.zeros(ComplexF32, ns * compact)
    terms = heisenberg_terms(lx, ly; J1 = 1.0, J2 = 0.0)
    emit(io, "[terms] diagonal bonds ", length(terms.diag), " flip terms ", length(terms.flip))
    for bond in terms.diag
        emit(io, "[terms] diagonal site_a ", bond.site_a, " site_b ", bond.site_b, " coeff ", bond.coeff)
    end
    for term in terms.flip
        emit(io, "[terms] flip n_flips ", term.n_flips, " sites ", term.flip_site,
            " values ", term.flip_value, " mask ", (term.mask_a, term.mask_b),
            " coeff ", (term.coeff_re, term.coeff_im))
    end

    emit(io, "[eo] loaded through unified package binding")
    diag = terms.diag
    flip = terms.flip
    status = GC.@preserve peps samples logpsi e_loc rows diag flip begin
        table = QnpepsElocTermTable(
            Int32(length(diag)),
            pointer(diag),
            Int32(length(flip)),
            pointer(flip),
        )
        inputs = FFI._ElocRunInputs(
            ;
            peps=reinterpret(CuPtr{Cvoid}, pointer(peps)),
            samples=pointer(samples),
            terms=table,
        )
        outputs = FFI._ElocRunOutputs(
            ;
            logpsi=pointer(logpsi),
            e_loc=pointer(e_loc),
            rows=reinterpret(CuPtr{Cvoid}, pointer(rows)),
        )
        workspace = FFI._ElocRunWorkspace(
            ;
            workspace=Ptr{Cvoid}(0),
            scratch=Ptr{Cvoid}(0),
            reference_energy=0.0,
            stream=Ptr{Cvoid}(0),
        )
        arguments = FFI._ElocRunArguments(
            ; config, n_samples=Int64(ns), inputs, outputs, workspace
        )
        FFI.eloc_run(arguments)
    end
    emit(io, "[eo] qnpeps_eloc_run status ", status)
    status == 0 || error("qnpeps_eloc_run returned status $status")
    CUDA.synchronize()

    host_logpsi = Array(logpsi)
    host_e_loc = Array(e_loc)
    host_rows = Array(rows)
    for j in 1:ns
        emit(io, "")
        emit(io, "---- sample ", j, " configuration ", Int.(configurations[j, :]), " ----")
        emitf(io, "[device] log|psi| %.16e phase %.16e", host_logpsi[2j - 1], host_logpsi[2j])
        emitf(io, "[host] cpu reference log|psi| %.16e phase %.16e",
            CPU_REFERENCE["eo_logabs"][j], CPU_REFERENCE["eo_phase"][j])
        emitf(io, "[compare] log|psi| difference %.6e phase difference %.6e",
            abs(host_logpsi[2j - 1] - CPU_REFERENCE["eo_logabs"][j]),
            abs(host_logpsi[2j] - CPU_REFERENCE["eo_phase"][j]))
        emitf(io, "[device] e_loc % .16e %+.16eim", host_e_loc[2j - 1], host_e_loc[2j])
        emitf(io, "[host] cpu reference e_loc % .16e %+.16eim",
            CPU_REFERENCE["eo_eloc_real"][j], CPU_REFERENCE["eo_eloc_imag"][j])
        emitf(io, "[compare] e_loc absolute difference %.6e",
            abs(complex(host_e_loc[2j - 1], host_e_loc[2j]) -
                complex(CPU_REFERENCE["eo_eloc_real"][j], CPU_REFERENCE["eo_eloc_imag"][j])))
        block = host_rows[((j - 1) * compact + 1):(j * compact)]
        show_vector(io, "[device] compact Ok row", block)
        checks = ComplexF64[]
        offset = 0
        for i in 0:1, c in 0:1
            w, s, e, n = proj_dims(lx, ly, dim_bond, i, c)
            slice = w * s * e * n
            spin = Int(configurations[j, i * ly + c + 1])
            site_wnse = permutedims(sites[(i, c)][:, :, :, :, spin + 1], (1, 4, 2, 3))
            push!(checks, sum(vec(site_wnse) .* ComplexF64.(block[(offset + 1):(offset + slice)])))
            offset += slice
        end
        show_vector(io, "[device] per site contraction of the site tensor against its Ok block", checks)
        emitf(io, "[device] Euler identity total %.16e expected %d", real(sum(checks)), lx * ly)
    end
end

section("minsr_gram") do io
    lx = 2
    ly = 2
    dim_bond = 1
    dim_phys = 2
    ns = 3
    relative_cut = 1.0e-3
    absolute_cut = 1.0e-8
    compact = Int(minsr_compact_count(; lx, ly, dim_bond, n_samples = ns, dim_phys))
    dense = Int(minsr_dense_count(; lx, ly, dim_bond, n_samples = ns, dim_phys))
    emit(io, "[geometry] lx ", lx, " ly ", ly, " dim_bond ", dim_bond, " dim_phys ", dim_phys)
    emit(io, "[geometry] compact_count ", compact, " dense_count ", dense, " n_samples ", ns)
    show_matrix(io, "[input] Ok compact rows", OK_ROWS_CPU)
    emit(io, "[input] sample spins per site")
    for j in 1:ns
        emit(io, "  sample ", j, " ", Int.(SPINS_CPU[j, :]))
    end

    logpsi_host = ComplexF64[-0.30+0.20im, -0.10-0.40im, -0.55+0.15im]
    e_loc_host = ComplexF64[-1.20+0.05im, -0.90-0.10im, -1.55+0.20im]
    logq_host = [-0.75, -0.20, -1.10]

    samples = CuArray(vec(permutedims(SPINS_CPU, (2, 1))))
    o_rows = CuArray(vec(permutedims(ComplexF32.(OK_ROWS_CPU), (2, 1))))
    gram = CUDA.zeros(ComplexF32, ns * ns)
    logpsi = CuArray(logpsi_host)
    e_loc = CuArray(e_loc_host)
    logq = CuArray(logq_host)

    gram_context = GramContext(; lx, ly, dim_bond, n_samples=ns, dim_phys)
    try
        footprint = gram_footprint(gram_context)
        emit(io, "[gram] context device bytes ", footprint.context_device_bytes,
            " geometry bytes ", footprint.geometry_device_bytes,
            " dense a bytes ", footprint.dense_a_device_bytes,
            " dense b bytes ", footprint.dense_b_device_bytes)
        raw_gram!(gram_context, gram, samples, o_rows)
        CUDA.synchronize()
    finally
        close(gram_context)
    end
    gram_host = permutedims(reshape(Array(gram), ns, ns), (2, 1))
    show_matrix(io, "[device] raw sector Gram", gram_host)

    expected_gram = zeros(ComplexF64, ns, ns)
    for s in 1:ns, u in 1:ns
        acc = zero(ComplexF64)
        for b in 1:(lx * ly)
            SPINS_CPU[s, b] == SPINS_CPU[u, b] || continue
            acc += conj(OK_ROWS_CPU[s, b]) * OK_ROWS_CPU[u, b]
        end
        expected_gram[s, u] = acc
    end
    show_matrix(io, "[host] cpu reference sector Gram", expected_gram)
    emitf(io, "[compare] Gram maximum absolute difference %.6e", maximum(abs.(gram_host - expected_gram)))

    result = minsr_direction(
        samples, logpsi, e_loc, logq, gram, o_rows;
        lx, ly, dim_bond, dim_phys, n_samples = ns, relative_cut, absolute_cut,
    )
    CUDA.synchronize()
    emitf(io, "[device] e_mean % .16e %+.16eim", real(result.e_mean), imag(result.e_mean))
    emitf(io, "[device] e_var %.16e", result.e_var)
    emitf(io, "[device] ess %.16e", result.ess)
    emitf(io, "[host] cpu reference e_mean % .16e %+.16eim",
        CPU_REFERENCE["minsr_e_mean_real"], CPU_REFERENCE["minsr_e_mean_imag"])
    emitf(io, "[host] cpu reference e_var %.16e", CPU_REFERENCE["minsr_e_var"])
    emitf(io, "[host] cpu reference ess %.16e", CPU_REFERENCE["minsr_ess"])
    emitf(io, "[compare] ess difference %.6e", abs(result.ess - CPU_REFERENCE["minsr_ess"]))
    emitf(io, "[compare] e_var relative difference %.6e",
        abs(result.e_var - CPU_REFERENCE["minsr_e_var"]) / CPU_REFERENCE["minsr_e_var"])
    theta = ComplexF64.(Array(result.theta_dot))
    show_vector(io, "[device] theta_dot", theta)
    show_vector(io, "[host] cpu reference theta_dot", CPU_REFERENCE["minsr_theta_dot"])
    emitf(io, "[compare] theta_dot maximum absolute difference %.6e",
        maximum(abs.(theta - CPU_REFERENCE["minsr_theta_dot"])))
    emitf(io, "[compare] theta_dot relative difference %.6e",
        norm(theta - CPU_REFERENCE["minsr_theta_dot"]) / norm(CPU_REFERENCE["minsr_theta_dot"]))
end

open(joinpath(OUT, "combined.log"), "w") do io
    for name in ("environment", "truncation_panels", "zipup_three_site", "eo_two_by_two", "minsr_gram")
        path = joinpath(OUT, name * ".txt")
        isfile(path) && write(io, read(path, String))
    end
end
println("wrote ", OUT)
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && walkthrough_examples()
