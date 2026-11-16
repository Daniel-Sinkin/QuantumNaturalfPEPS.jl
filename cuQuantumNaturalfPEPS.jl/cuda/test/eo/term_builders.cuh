#ifndef QNPEPS_TEST_EO_TERM_BUILDERS_CUH
#define QNPEPS_TEST_EO_TERM_BUILDERS_CUH

#include "dans_qnpeps_eloc.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <vector>

namespace eo_test
{
using Complex = std::complex<double>;

inline constexpr int k_spin_states{2};

class TermTableBuilder
{
  public:
    auto add_diagonal(const std::vector<int>& sites, const std::vector<Complex>& values) -> void
    {
        auto term = QnpepsElocDiagonalTerm{};
        term.value_offset = static_cast<std::int64_t>(diagonal_values_.size());
        term.site_count = static_cast<std::int32_t>(sites.size());
        term.state_count = static_cast<std::int32_t>(values.size());
        std::copy(sites.begin(), sites.end(), term.site);
        diagonal_.push_back(term);
        for (const auto& value : values)
            diagonal_values_.push_back({value.real(), value.imag()});
    }

    auto add_piece(
        const std::vector<int>& sites,
        const std::vector<std::uint8_t>& active,
        const std::vector<Complex>& coefficients,
        const std::vector<Complex>& weights
    ) -> void
    {
        auto piece = QnpepsElocOffDiagonalPiece{};
        piece.state_offset = static_cast<std::int64_t>(piece_active_.size());
        piece.weight_offset = static_cast<std::int64_t>(piece_weights_.size());
        piece.site_count = static_cast<std::int32_t>(sites.size());
        piece.state_count = static_cast<std::int32_t>(active.size());
        std::copy(sites.begin(), sites.end(), piece.site);
        piece_.push_back(piece);
        piece_active_.insert(piece_active_.end(), active.begin(), active.end());
        for (const auto& value : coefficients)
            piece_coefficients_.push_back({value.real(), value.imag()});
        for (const auto& value : weights)
            piece_weights_.push_back({value.real(), value.imag()});
    }

    auto add_zz(int site_a, int site_b, double coupling) -> void
    {
        add_diagonal({site_a, site_b}, {coupling, -coupling, -coupling, coupling});
    }

    auto add_z(int site, double field) -> void { add_diagonal({site}, {field, -field}); }

    auto add_reflection(
        const std::vector<int>& flipped,
        const std::vector<int>& spectators,
        Complex coefficient,
        int mask_a,
        int mask_b
    ) -> void
    {
        auto sites = flipped;
        sites.insert(sites.end(), spectators.begin(), spectators.end());
        const auto site_count = sites.size();
        const auto state_count = std::size_t{1} << site_count;
        auto active = std::vector<std::uint8_t>(state_count, 1);
        auto coefficients = std::vector<Complex>(state_count, coefficient);
        auto weights = std::vector<Complex>(state_count * site_count * k_spin_states, 0.0);
        for (auto state = std::size_t{0}; state < state_count; ++state)
        {
            const auto value_at = [&](std::size_t position)
            { return static_cast<int>((state >> (site_count - 1 - position)) & 1u); };
            const auto position_of = [&](int site)
            {
                return static_cast<std::size_t>(
                    std::find(sites.begin(), sites.end(), site) - sites.begin()
                );
            };
            if (mask_a >= 0 and value_at(position_of(mask_a)) == value_at(position_of(mask_b)))
            {
                active[state] = 0;
                coefficients[state] = 0.0;
            }
            for (auto position = std::size_t{0}; position < site_count; ++position)
            {
                const auto own = value_at(position);
                const auto target = position < flipped.size() ? 1 - own : own;
                const auto slot = (state * site_count + position) * k_spin_states;
                weights[slot + static_cast<std::size_t>(target)] = active[state] != 0 ? 1.0 : 0.0;
            }
        }
        add_piece(sites, active, coefficients, weights);
    }

    auto add_one_site_flip(int site, Complex coefficient) -> void
    {
        add_reflection({site}, {}, coefficient, -1, -1);
    }

    auto add_exchange(int site_a, int site_b, Complex coefficient) -> void
    {
        add_reflection({site_a, site_b}, {}, coefficient, site_a, site_b);
    }

    auto add_heisenberg_bond(int site_a, int site_b, double coupling) -> void
    {
        add_zz(site_a, site_b, coupling);
        add_exchange(site_a, site_b, 2.0 * coupling);
    }

    [[nodiscard]] auto table() const -> QnpepsElocTermTable
    {
        auto table = QnpepsElocTermTable{};
        table.struct_size = sizeof(QnpepsElocTermTable);
        table.dim_phys = k_spin_states;
        table.diagonal_count = static_cast<std::int32_t>(diagonal_.size());
        table.piece_count = static_cast<std::int32_t>(piece_.size());
        table.diagonal = diagonal_.empty() ? nullptr : diagonal_.data();
        table.piece = piece_.empty() ? nullptr : piece_.data();
        table.diagonal_value_count = static_cast<std::int64_t>(diagonal_values_.size());
        table.diagonal_values = diagonal_values_.empty() ? nullptr : diagonal_values_.data();
        table.piece_state_count = static_cast<std::int64_t>(piece_active_.size());
        table.piece_active = piece_active_.empty() ? nullptr : piece_active_.data();
        table.piece_coefficients =
            piece_coefficients_.empty() ? nullptr : piece_coefficients_.data();
        table.piece_weight_count = static_cast<std::int64_t>(piece_weights_.size());
        table.piece_weights = piece_weights_.empty() ? nullptr : piece_weights_.data();
        return table;
    }

    [[nodiscard]] auto product_energy(
        const std::uint8_t* sample, const std::vector<double>& amplitude
    ) const -> Complex
    {
        auto energy = Complex{};
        for (const auto& term : diagonal_)
        {
            const auto value = diagonal_values_[static_cast<std::size_t>(
                term.value_offset + state_of(term.site, term.site_count, sample)
            )];
            energy += Complex{value.re, value.im};
        }
        for (const auto& piece : piece_)
        {
            const auto state = state_of(piece.site, piece.site_count, sample);
            const auto index = static_cast<std::size_t>(piece.state_offset + state);
            if (piece_active_[index] == 0) continue;
            auto ratio = Complex{piece_coefficients_[index].re, piece_coefficients_[index].im};
            for (auto position = 0; position < piece.site_count; ++position)
            {
                const auto site = piece.site[position];
                const auto own =
                    amplitude[static_cast<std::size_t>(k_spin_states * site + sample[site])];
                const auto base =
                    piece.weight_offset + (state * piece.site_count + position) * k_spin_states;
                auto factor = Complex{};
                for (auto value = 0; value < k_spin_states; ++value)
                {
                    const auto& weight = piece_weights_[static_cast<std::size_t>(base + value)];
                    factor += Complex{weight.re, weight.im}
                              * amplitude[static_cast<std::size_t>(k_spin_states * site + value)]
                              / own;
                }
                ratio *= factor;
            }
            energy += ratio;
        }
        return energy;
    }

  private:
    [[nodiscard]] static auto state_of(
        const std::int32_t* sites, std::int32_t count, const std::uint8_t* sample
    ) -> std::int64_t
    {
        auto state = std::int64_t{0};
        for (auto position = 0; position < count; ++position)
            state = state * k_spin_states + sample[sites[position]];
        return state;
    }

    std::vector<QnpepsElocDiagonalTerm> diagonal_{};
    std::vector<QnpepsElocComplex> diagonal_values_{};
    std::vector<QnpepsElocOffDiagonalPiece> piece_{};
    std::vector<std::uint8_t> piece_active_{};
    std::vector<QnpepsElocComplex> piece_coefficients_{};
    std::vector<QnpepsElocComplex> piece_weights_{};
};
}

#endif
