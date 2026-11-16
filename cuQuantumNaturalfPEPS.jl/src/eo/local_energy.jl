function local_energies(
    peps::CuVector{ComplexF32},
    config::QnpepsElocConfig,
    terms::OperatorTerms,
    samples::CuVector{UInt8},
)
    sites = Int(config.lx) * Int(config.ly)
    n_samples, partial_sample = divrem(length(samples), sites)
    partial_sample == 0 ||
        throw(ArgumentError("sample buffer length must be a multiple of lx * ly"))
    logpsi = CUDA.zeros(Float64, 2 * n_samples)
    e_loc = CUDA.zeros(Float64, 2 * n_samples)
    stream = CUDA.stream()
    buffers = _eloc_term_buffers(terms, config)
    status = GC.@preserve peps samples logpsi e_loc buffers begin
        inputs = FFI._ElocRunInputs(;
            peps=reinterpret(CuPtr{Cvoid}, pointer(peps)),
            samples=pointer(samples),
            terms=_eloc_term_table(buffers),
        )
        outputs = FFI._ElocRunOutputs(;
            logpsi=pointer(logpsi),
            e_loc=pointer(e_loc),
            rows=CuPtr{Cvoid}(0),
        )
        workspace = FFI._ElocRunWorkspace(;
            workspace=C_NULL,
            scratch=C_NULL,
            reference_energy=0.0,
            stream=Ptr{Cvoid}(stream.handle),
        )
        FFI.eloc_run(
            FFI._ElocRunArguments(; config, n_samples=Int64(n_samples), inputs, outputs, workspace),
        )
    end
    _check(; status, what="qnpeps_eloc_run")
    return (logpsi=reinterpret(ComplexF64, logpsi), e_loc=reinterpret(ComplexF64, e_loc))
end
