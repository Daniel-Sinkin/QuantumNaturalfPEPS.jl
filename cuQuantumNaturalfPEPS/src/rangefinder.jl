using CUDA

function batched_rangefinder(
    input::CuArray{ComplexF32,3},
    rank::Integer;
    seed::Integer=0,
)::Tuple{CuArray{ComplexF32,3},CuArray{ComplexF32,3}}
    rows, cols, batch = size(input)
    if !(1 <= rank <= min(rows, cols))
        throw(
            ArgumentError("rank must be in 1:min(rows, cols) (rank=$rank, rows=$rows, cols=$cols)"),
        )
    end
    qs = CUDA.zeros(ComplexF32, rows, rank, batch)
    rs = CUDA.zeros(ComplexF32, rank, cols, batch)
    scratch_bytes = _batched_rangefinder_scratch_bytes(; rows, cols, rank, batch)
    scratch = CUDA.zeros(UInt8, scratch_bytes)
    _ffi_batched_rangefinder(;
        input=pointer(input),
        rows,
        cols,
        rank,
        batch,
        input_stride=rows * cols,
        seed,
        q_out=pointer(qs),
        q_stride=rows * rank,
        r_out=pointer(rs),
        r_stride=rank * cols,
        scratch=pointer(scratch),
        scratch_bytes,
    )
    return qs, rs
end
