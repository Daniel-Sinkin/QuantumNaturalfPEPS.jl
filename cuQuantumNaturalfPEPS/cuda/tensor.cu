#include "arena_cursor.cuh"
#include "tensor.cuh"

namespace qnpeps
{
auto alloc(ArenaCursor& arena, const Shape& dim) -> DeviceTensor
{
    DeviceTensor tensor{};
    tensor.dim = dim;
    tensor.d = arena.take<cuFloatComplex>(tensor.num_elems());
    return tensor;
}

auto free(DeviceTensor&) -> void {}
}
