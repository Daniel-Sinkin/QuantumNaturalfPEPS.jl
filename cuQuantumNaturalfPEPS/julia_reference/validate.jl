using LinearAlgebra
using Random
using Test

include(joinpath(@__DIR__, "JuliaReference.jl"))
using .JuliaReference

function random_peps(
    lx::Int,
    ly::Int,
    dim_phys::Int,
    dim_bond::Int;
    seed::Int,
    element_type=ComplexF64,
)
    T = element_type
    T <: Complex || throw(ArgumentError("element_type must be complex"))
    rng = MersenneTwister(seed)
    peps = Matrix{Array{T,5}}(undef, lx, ly)
    for row in 1:lx, col in 1:ly
        up = row == 1 ? 1 : dim_bond
        right = col == ly ? 1 : dim_bond
        down = row == lx ? 1 : dim_bond
        left = col == 1 ? 1 : dim_bond
        peps[row, col] = convert(T, 0.17) .* randn(
            rng,
            T,
            dim_phys,
            up,
            right,
            down,
            left,
        )
    end
    return peps
end

function configuration_from_code(code::Integer, lx::Int, ly::Int, dim_phys::Int)
    configuration = Matrix{UInt8}(undef, lx, ly)
    value = Int(code)
    for site0 in 0:(lx*ly-1)
        row, col = fld(site0, ly) + 1, mod(site0, ly) + 1
        configuration[row, col] = UInt8(mod(value, dim_phys))
        value = fld(value, dim_phys)
    end
    return configuration
end

function all_configurations(lx::Int, ly::Int, dim_phys::Int)
    return [
        configuration_from_code(code, lx, ly, dim_phys) for
        code in 0:(dim_phys^(lx*ly)-1)
    ]
end

function exact_amplitude(peps::AbstractMatrix, configuration::AbstractMatrix)
    lx, ly = size(peps)
    size(configuration) == (lx, ly) || throw(DimensionMismatch())
    dim_bond = size(peps[1, 1], 3)
    horizontal_count = lx * (ly - 1)
    vertical_count = (lx - 1) * ly
    ranges = ntuple(_ -> 1:dim_bond, horizontal_count + vertical_count)
    amplitude = 0.0 + 0.0im

    horizontal_index(row, col) = (row - 1) * (ly - 1) + col
    vertical_index(row, col) = horizontal_count + (row - 1) * ly + col
    for assignment in Iterators.product(ranges...)
        product_value = 1.0 + 0.0im
        for row in 1:lx, col in 1:ly
            up = row == 1 ? 1 : assignment[vertical_index(row - 1, col)]
            right = col == ly ? 1 : assignment[horizontal_index(row, col)]
            down = row == lx ? 1 : assignment[vertical_index(row, col)]
            left = col == 1 ? 1 : assignment[horizontal_index(row, col - 1)]
            product_value *= peps[row, col][
                Int(configuration[row, col]) + 1,
                up,
                right,
                down,
                left,
            ]
        end
        amplitude += product_value
    end
    return ComplexF64(amplitude)
end

function exact_factorizer(seed::Integer=777)
    return RangefinderFactorizer(
        seed=UInt64(seed),
        force_fallback=true,
        recovery=:householder,
        effective_rank_rtol=1.0e-12,
    )
end

@testset "ordered contractions and fixed-rank rangefinder" begin
    tensor_a = reshape(ComplexF64.(1:24), 2, 3, 4)
    tensor_b = reshape(complex.(1:60, 60:-1:1), 5, 4, 3)
    plan = contract_plan(tensor_a, (3, 2), tensor_b, (2, 3))
    @test plan.permutation_a == (1, 3, 2)
    @test plan.permutation_b == (2, 3, 1)
    @test plan.result_shape == (2, 5)
    expected = zeros(ComplexF64, 2, 5)
    for left in 1:2, right in 1:5, axis_three in 1:4, axis_two in 1:3
        expected[left, right] +=
            tensor_a[left, axis_two, axis_three] * tensor_b[right, axis_three, axis_two]
    end
    @test contract_arrays(tensor_a, (3, 2), tensor_b, (2, 3)) == expected

    rng = MersenneTwister(3)
    matrix = randn(rng, ComplexF64, 8, 5)
    normal = batched_rangefinder(
        matrix,
        3;
        factorizer=RangefinderFactorizer(seed=UInt64(19)),
    )
    @test normal.diagnostic.status == :ok
    @test size(normal.q) == (8, 3)
    @test size(normal.r) == (3, 5)
    @test all(isfinite, normal.q)
    @test all(isfinite, normal.r)

    recovered = batched_rangefinder(matrix, 5; factorizer=exact_factorizer(19))
    @test recovered.diagnostic.status == :householder_recovered
    @test recovered.diagnostic.requested_rank == 5
    @test recovered.diagnostic.capped_rank == 5
    @test recovered.diagnostic.effective_rank == 5
    @test recovered.diagnostic.fallback_used
    @test isapprox(recovered.q * recovered.r, matrix; rtol=2.0e-12, atol=2.0e-12)

    rank_one = ComplexF64.(collect(1:6)) * transpose(ComplexF64.(collect(1:5)))
    refused = batched_rangefinder(rank_one, 3; factorizer=exact_factorizer(23))
    @test refused.q === nothing
    @test refused.r === nothing
    @test refused.diagnostic.status == :unsupported_fixed_rank
    @test refused.diagnostic.effective_rank == 1
    @test_throws FixedRankRangefinderError factorize_matrix(exact_factorizer(23), rank_one, 3)
end

@testset "generic MPO-MPS zip-up" begin
    rng = MersenneTwister(9)
    mpo = [randn(rng, ComplexF64, 1, 2, 3, 1)]
    mps = [randn(rng, ComplexF64, 1, 2, 1)]
    result, log_gauge = zipup_mpo_mps(
        mpo,
        mps;
        maxdim=3,
        factorizer=exact_factorizer(29),
    )
    expected = [
        sum(mps[1][1, physical_in, 1] * mpo[1][1, physical_in, physical_out, 1] for
            physical_in in 1:2) for physical_out in 1:3
    ]
    @test length(result) == 1
    @test size(result[1]) == (1, 3, 1)
    @test isapprox(vec(result[1]) .* exp(log_gauge), expected; rtol=2.0e-12, atol=2.0e-12)
end

@testset "double-layer stack" begin
    peps = random_peps(3, 2, 2, 2; seed=11)
    stack = build_double_layer_stack(
        peps;
        chi_s=4,
        chi_dl=4,
        factorizer=exact_factorizer(31),
    )
    @test stack.lx == 3
    @test stack.ly == 2
    @test stack.chi_s == 4
    @test stack.chi_dl == 4
    @test length(stack.environments) == 2
    @test length(stack.cumulative_row_logs) == 2
    @test all(isfinite, stack.cumulative_row_logs)
    for environment in stack.environments
        @test length(environment) == 2
        @test size(environment[1], 1) == 1
        @test size(environment[end], 4) == 1
        @test size(environment[1], 4) == size(environment[2], 1)
    end
end

@testset "sampler proposals and mode semantics" begin
    peps = random_peps(2, 2, 2, 2; seed=17)
    stack = build_double_layer_stack(
        peps;
        chi_s=8,
        chi_dl=4,
        factorizer=exact_factorizer(41),
    )
    context = SamplerContext(
        peps,
        stack;
        chi_s=8,
        sampling_mode=:fast,
        chi_c=8,
        seed=53,
        factorizer=exact_factorizer(43),
    )
    configurations = all_configurations(2, 2, 2)
    proposal = [
        exp(proposal_log_probability(context, configuration).log_probability) for
        configuration in configurations
    ]
    amplitudes = exact_amplitude.(Ref(peps), configurations)
    born = abs2.(amplitudes)
    born ./= sum(born)
    @test isapprox(sum(proposal), 1.0; rtol=2.0e-11, atol=2.0e-11)
    @test isapprox(proposal, born; rtol=2.0e-9, atol=2.0e-11)

    rectangular = random_peps(3, 2, 2, 2; seed=29)
    truncated_stack = build_double_layer_stack(
        rectangular;
        chi_s=1,
        chi_dl=2,
        factorizer=exact_factorizer(59),
    )
    fast = SamplerContext(
        rectangular,
        truncated_stack;
        chi_s=1,
        sampling_mode=:fast,
        chi_c=4,
        seed=61,
        factorizer=exact_factorizer(67),
    )
    full = SamplerContext(
        rectangular,
        truncated_stack;
        chi_s=1,
        sampling_mode=:full,
        chi_c=4,
        seed=61,
        factorizer=exact_factorizer(67),
    )
    rectangular_configurations = all_configurations(3, 2, 2)
    fast_proposal = [
        exp(proposal_log_probability(fast, configuration).log_probability) for
        configuration in rectangular_configurations
    ]
    full_proposal = [
        exp(proposal_log_probability(full, configuration).log_probability) for
        configuration in rectangular_configurations
    ]
    @test isapprox(sum(fast_proposal), 1.0; rtol=2.0e-10, atol=2.0e-10)
    @test isapprox(sum(full_proposal), 1.0; rtol=2.0e-10, atol=2.0e-10)
    @test maximum(abs.(fast_proposal .- full_proposal)) > 1.0e-8

    first_run = ctx_sample_run(fast, 7; batch_base=5)
    replay = ctx_sample_run(
        SamplerContext(
            rectangular,
            truncated_stack;
            chi_s=1,
            sampling_mode=:fast,
            chi_c=4,
            seed=61,
            factorizer=exact_factorizer(67),
        ),
        7;
        batch_base=5,
    )
    @test first_run == replay
    @test all(configuration -> size(configuration) == (3, 2), first_run.configs)
    @test all(isfinite, first_run.log_prob_config)
    @test all(isfinite, first_run.log_gauge)
end
