#ifndef QNPEPS_ELOC_TERM_TABLE_CUH
#define QNPEPS_ELOC_TERM_TABLE_CUH

#include "core/types.cuh"
#include "dans_qnpeps_eloc.h"

#include <vector>

namespace qnpeps::eo
{
template <typename T>
[[nodiscard]] inline auto copy_pool(const T* pool, i64 count) -> std::vector<T>
{
    if (count < 1) return {};
    return std::vector<T>(pool, pool + count);
}

template <typename T>
[[nodiscard]] inline auto pool_data(const std::vector<T>& pool) noexcept -> const T*
{
    return pool.empty() ? nullptr : pool.data();
}

class OwnedTermTable
{
  public:
    OwnedTermTable() = default;

    explicit OwnedTermTable(const QnpepsElocTermTable& source)
        : dim_phys_(source.dim_phys), diagonal_(copy_pool(source.diagonal, source.diagonal_count)),
          piece_(copy_pool(source.piece, source.piece_count)),
          diagonal_values_(copy_pool(source.diagonal_values, source.diagonal_value_count)),
          piece_active_(copy_pool(source.piece_active, source.piece_state_count)),
          piece_coefficients_(copy_pool(source.piece_coefficients, source.piece_state_count)),
          piece_weights_(copy_pool(source.piece_weights, source.piece_weight_count))
    {
    }

    [[nodiscard]] auto view() const noexcept -> QnpepsElocTermTable
    {
        return QnpepsElocTermTable{
            .struct_size = sizeof(QnpepsElocTermTable),
            .dim_phys = dim_phys_,
            .diagonal_count = static_cast<i32>(diagonal_.size()),
            .piece_count = static_cast<i32>(piece_.size()),
            .diagonal = pool_data(diagonal_),
            .piece = pool_data(piece_),
            .diagonal_value_count = static_cast<i64>(diagonal_values_.size()),
            .diagonal_values = pool_data(diagonal_values_),
            .piece_state_count = static_cast<i64>(piece_active_.size()),
            .piece_active = pool_data(piece_active_),
            .piece_coefficients = pool_data(piece_coefficients_),
            .piece_weight_count = static_cast<i64>(piece_weights_.size()),
            .piece_weights = pool_data(piece_weights_),
        };
    }

  private:
    i32 dim_phys_{};
    std::vector<QnpepsElocDiagonalTerm> diagonal_{};
    std::vector<QnpepsElocOffDiagonalPiece> piece_{};
    std::vector<QnpepsElocComplex> diagonal_values_{};
    std::vector<u8> piece_active_{};
    std::vector<QnpepsElocComplex> piece_coefficients_{};
    std::vector<QnpepsElocComplex> piece_weights_{};
};
}

#endif
