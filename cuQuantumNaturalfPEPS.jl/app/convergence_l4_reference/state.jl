function convergence_site_tensor(tensor, row::Int, column::Int, lx::Int, ly::Int)
    selectors = Any[
        column == 1 ? 1 : Colon(),
        row == lx ? 1 : Colon(),
        column == ly ? 1 : Colon(),
        row == 1 ? 1 : Colon(),
        Colon(),
    ]
    labels = Tuple{Symbol,Int,Int}[]
    column > 1 && push!(labels, (:horizontal, row, column - 1))
    row < lx && push!(labels, (:vertical, row, column))
    column < ly && push!(labels, (:horizontal, row, column))
    row > 1 && push!(labels, (:vertical, row - 1, column))
    push!(labels, (:physical, row, column))
    return ComplexF64.(view(tensor, selectors...)), labels
end

function convergence_extend(frontier, frontier_labels, tensor, tensor_labels)
    contracted = [label for label in tensor_labels if label in frontier_labels]
    a_contract = [1 + only(findall(==(label), frontier_labels)) for label in contracted]
    b_contract = [only(findall(==(label), tensor_labels)) for label in contracted]
    a_keep = [axis for axis in 1:ndims(frontier) if axis ∉ a_contract]
    b_keep = [axis for axis in 1:ndims(tensor) if axis ∉ b_contract]
    a_outer = prod((size(frontier, axis) for axis in a_keep); init=1)
    b_outer = prod((size(tensor, axis) for axis in b_keep); init=1)
    contracted_count = prod((size(frontier, axis) for axis in a_contract); init=1)
    left = reshape(permutedims(frontier, (a_keep..., a_contract...)), a_outer, contracted_count)
    right = reshape(permutedims(tensor, (b_contract..., b_keep...)), contracted_count, b_outer)
    dims = (
        map(axis -> size(frontier, axis), a_keep)...,
        map(axis -> size(tensor, axis), b_keep)...,
    )
    product = reshape(left * right, dims)
    physical_keep = only(findall(axis -> tensor_labels[axis][1] == :physical, b_keep))
    physical_axis = length(a_keep) + physical_keep
    remaining_axes = Int[2:length(a_keep)...]
    append!(
        remaining_axes,
        (length(a_keep) + index for index in eachindex(b_keep) if index != physical_keep),
    )
    reordered = permutedims(product, (1, physical_axis, remaining_axes...))
    updated = reshape(
        reordered,
        size(frontier, 1) * size(tensor, b_keep[physical_keep]),
        (size(reordered, axis) for axis in 3:ndims(reordered))...,
    )
    remaining_a = [frontier_labels[axis-1] for axis in a_keep if axis != 1]
    remaining_b = [
        tensor_labels[axis] for axis in b_keep if tensor_labels[axis][1] != :physical
    ]
    scale = maximum(abs, updated)
    isfinite(scale) && scale > 0 || error("exact contraction scale is invalid")
    updated ./= scale
    return updated, vcat(remaining_a, remaining_b)
end

function convergence_exact_state(device_tensors)
    lx, ly = size(device_tensors)
    frontier = ComplexF64[1]
    labels = Tuple{Symbol,Int,Int}[]
    for row in 1:lx, column in 1:ly
        tensor, tensor_labels =
            convergence_site_tensor(device_tensors[row, column], row, column, lx, ly)
        frontier, labels = convergence_extend(frontier, labels, tensor, tensor_labels)
    end
    isempty(labels) || error("exact state contraction left virtual edges open")
    state = vec(frontier)
    state ./= norm(state)
    return state
end

function convergence_overlap(tensors, reference_device_tensors)
    device = map(tensor -> permutedims(tensor, (5, 4, 3, 2, 1)), tensors)
    observed = convergence_exact_state(device)
    reference = convergence_exact_state(reference_device_tensors)
    return abs(dot(reference, observed))
end
