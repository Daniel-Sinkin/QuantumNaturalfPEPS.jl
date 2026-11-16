#include "env_build/terms.cuh"
#include "term_builders.cuh"
#include "term_validation.cuh"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace
{
using qn_eloc::env::Bucket;
using qn_eloc::env::TermPlan;
using qnpeps::eo::TermValidation;

auto g_failures = 0;

auto record(const std::string& name, bool passed) -> void
{
    std::printf("[term_plan_test] case=%s status=%s\n", name.c_str(), passed ? "PASS" : "FAIL");
    if (not passed) ++g_failures;
}

auto config(int lx, int ly) -> QnpepsElocConfig
{
    return QnpepsElocConfig{sizeof(QnpepsElocConfig), lx, ly, 2, 2, 4, 4, 0, 0.0, 0.0};
}

auto validate(const QnpepsElocTermTable& table, int lx = 2, int ly = 3) -> TermValidation
{
    return qnpeps::eo::validate_terms(lx, ly, 2, &table);
}

auto plan(const eo_test::TermTableBuilder& terms, TermPlan& out, int lx = 2, int ly = 3) -> bool
{
    const auto table = terms.table();
    if (validate(table, lx, ly) != TermValidation::ok) return false;
    return qn_eloc::env::plan_terms(config(lx, ly), table, out);
}

auto first_piece(const std::vector<qn_eloc::env::TermPiece>& pieces) -> qn_eloc::env::TermPiece
{
    return pieces.empty() ? qn_eloc::env::TermPiece{} : pieces.front();
}

auto heisenberg(int site_a, int site_b) -> eo_test::TermTableBuilder
{
    auto terms = eo_test::TermTableBuilder{};
    terms.add_heisenberg_bond(site_a, site_b, 1.0);
    return terms;
}

auto z_times_x(int site_a, int site_b) -> eo_test::TermTableBuilder
{
    auto terms = eo_test::TermTableBuilder{};
    terms.add_piece(
        {site_a, site_b},
        {1, 1, 1, 1},
        {1.0, 1.0, -1.0, -1.0},
        {1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0}
    );
    return terms;
}

auto run_validation_cases() -> void
{
    const auto valid = heisenberg(0, 1);
    record("validate_heisenberg", validate(valid.table()) == TermValidation::ok);
    record(
        "validate_null",
        qnpeps::eo::validate_terms(2, 3, 2, nullptr) == TermValidation::null_argument
    );
    const auto mutate = [&](const std::string& name, TermValidation expected, auto change)
    {
        auto table = valid.table();
        auto pieces =
            std::vector<QnpepsElocOffDiagonalPiece>(table.piece, table.piece + table.piece_count);
        auto diagonal = std::vector<QnpepsElocDiagonalTerm>(
            table.diagonal, table.diagonal + table.diagonal_count
        );
        auto active = std::vector<std::uint8_t>(
            table.piece_active, table.piece_active + table.piece_state_count
        );
        auto coefficients = std::vector<QnpepsElocComplex>(
            table.piece_coefficients, table.piece_coefficients + table.piece_state_count
        );
        auto weights = std::vector<QnpepsElocComplex>(
            table.piece_weights, table.piece_weights + table.piece_weight_count
        );
        table.piece = pieces.data();
        table.diagonal = diagonal.data();
        table.piece_active = active.data();
        table.piece_coefficients = coefficients.data();
        table.piece_weights = weights.data();
        change(table, pieces, diagonal, active, coefficients, weights);
        record(name, validate(table) == expected);
    };
    const auto bad = TermValidation::bad_config;
    mutate(
        "reject_struct_size",
        TermValidation::bad_version,
        [](auto& t, auto&, auto&, auto&, auto&, auto&) { t.struct_size = 0; }
    );
    mutate(
        "reject_dim_phys", bad, [](auto& t, auto&, auto&, auto&, auto&, auto&) { t.dim_phys = 3; }
    );
    mutate(
        "reject_site_range",
        bad,
        [](auto&, auto& p, auto&, auto&, auto&, auto&) { p[0].site[1] = 6; }
    );
    mutate(
        "reject_negative_site",
        bad,
        [](auto&, auto&, auto& d, auto&, auto&, auto&) { d[0].site[0] = -1; }
    );
    mutate(
        "reject_duplicate_site",
        bad,
        [](auto&, auto& p, auto&, auto&, auto&, auto&) { p[0].site[1] = 0; }
    );
    mutate(
        "reject_site_count_zero",
        bad,
        [](auto&, auto& p, auto&, auto&, auto&, auto&) { p[0].site_count = 0; }
    );
    mutate(
        "reject_site_count_five",
        bad,
        [](auto&, auto&, auto& d, auto&, auto&, auto&) { d[0].site_count = 5; }
    );
    mutate(
        "reject_state_count",
        bad,
        [](auto&, auto& p, auto&, auto&, auto&, auto&) { p[0].state_count = 3; }
    );
    mutate(
        "reject_value_extent",
        bad,
        [](auto&, auto&, auto& d, auto&, auto&, auto&) { d[0].value_offset = 1; }
    );
    mutate(
        "reject_state_extent",
        bad,
        [](auto&, auto& p, auto&, auto&, auto&, auto&) { p[0].state_offset = 1; }
    );
    mutate(
        "reject_weight_extent",
        bad,
        [](auto&, auto& p, auto&, auto&, auto&, auto&) { p[0].weight_offset = 2; }
    );
    mutate(
        "reject_negative_offset",
        bad,
        [](auto&, auto& p, auto&, auto&, auto&, auto&) { p[0].weight_offset = -1; }
    );
    mutate("reject_active_flag", bad, [](auto&, auto&, auto&, auto& a, auto&, auto&) { a[1] = 2; });
    mutate(
        "reject_all_inactive",
        bad,
        [](auto&, auto&, auto&, auto& a, auto&, auto&) { a.assign(a.size(), 0); }
    );
    mutate(
        "reject_nan_coefficient",
        bad,
        [](auto&, auto&, auto&, auto&, auto& c, auto&)
        { c[1].re = std::numeric_limits<double>::quiet_NaN(); }
    );
    mutate(
        "reject_zero_weights",
        bad,
        [](auto&, auto&, auto&, auto&, auto&, auto& w)
        {
            w[4] = {0.0, 0.0};
            w[5] = {0.0, 0.0};
        }
    );
    mutate(
        "reject_null_pool",
        TermValidation::null_argument,
        [](auto& t, auto&, auto&, auto&, auto&, auto&) { t.piece_weights = nullptr; }
    );
    mutate(
        "reject_negative_count",
        bad,
        [](auto& t, auto&, auto&, auto&, auto&, auto&) { t.piece_count = -1; }
    );
}

auto run_plan_cases() -> void
{
    {
        auto out = TermPlan{};
        const auto ok = plan(heisenberg(0, 1), out);
        const auto piece = first_piece(out.original);
        const auto expected_targets = std::vector<std::uint8_t>{0, 0, 1, 0, 0, 1, 1, 1};
        record(
            "plan_horizontal_heisenberg",
            ok and out.original.size() == 1 and out.transposed.empty()
                and piece.bucket == Bucket::horizontal and piece.is_one_hot
                and piece.moving_count == 2 and piece.active_slot == 0
                and out.targets == expected_targets and out.diagonal.size() == 1
        );
    }
    {
        auto out = TermPlan{};
        const auto ok = plan(heisenberg(0, 3), out);
        const auto piece = first_piece(out.transposed);
        record(
            "plan_vertical_transposed",
            ok and out.original.empty() and piece.pass == 1 and piece.bucket == Bucket::horizontal
                and piece.local_state.site[0] == 0 and piece.local_state.site[1] == 1
        );
    }
    {
        auto out = TermPlan{};
        const auto ok = plan(heisenberg(4, 0), out);
        record(
            "plan_diagonal_fourbody",
            ok and out.original.size() == 1 and out.original.front().bucket == Bucket::fourbody
        );
    }
    {
        auto out = TermPlan{};
        const auto ok = plan(heisenberg(0, 2), out);
        record(
            "plan_longer_horizontal",
            ok and out.original.size() == 1
                and out.original.front().bucket == Bucket::longer_horizontal
        );
    }
    {
        auto out = TermPlan{};
        record("plan_reject_knight", not plan(heisenberg(0, 5), out));
        record(
            "routable_rejects_knight", not qnpeps::eo::is_routable(2, 3, heisenberg(0, 5).table())
        );
        record("routable_accepts_bond", qnpeps::eo::is_routable(2, 3, heisenberg(0, 1).table()));
    }
    {
        auto terms = eo_test::TermTableBuilder{};
        terms.add_piece(
            {0, 5},
            {1, 1, 1, 1},
            {1.0, 1.0, 1.0, 1.0},
            {2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0}
        );
        auto out = TermPlan{};
        const auto ok = plan(terms, out);
        const auto piece = first_piece(out.original);
        auto is_folded = ok and out.original.size() == 1 and out.coefficients.size() == 4;
        for (const auto& coefficient : out.coefficients)
            is_folded = is_folded and coefficient.x == 2.0 and coefficient.y == 0.0;
        record(
            "plan_scaled_spectator_folds_into_coefficient",
            is_folded and piece.moving_count == 1 and piece.moving[0] == 5 and piece.is_one_hot
        );
    }
    {
        auto out = TermPlan{};
        const auto ok = plan(z_times_x(0, 5), out);
        const auto piece = first_piece(out.original);
        record(
            "plan_spectator_routes_on_moving_site",
            ok and piece.moving_count == 1 and piece.moving[0] == 5 and not piece.is_moving[0]
                and piece.is_moving[1] and piece.active_slot < 0
                and qn_eloc::env::support_position(piece, 0) == qn_eloc::env::k_keep_position
                and qn_eloc::env::support_position(piece, 5) == 1
        );
    }
    {
        auto terms = eo_test::TermTableBuilder{};
        terms.add_piece(
            {0, 1},
            {1, 1, 1, 1},
            {1.0, 1.0, 1.0, 1.0},
            {0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, -1.0}
        );
        auto out = TermPlan{};
        const auto ok = plan(terms, out);
        record(
            "plan_weighted_piece",
            ok and out.original.size() == 1 and not out.original.front().is_one_hot
                and out.targets.empty()
        );
    }
    {
        auto terms = eo_test::TermTableBuilder{};
        terms.add_one_site_flip(2, 0.5);
        terms.add_piece({3}, {1, 0}, {1.0, 0.0}, {0.0, 1.0, 1.0, 0.0});
        auto out = TermPlan{};
        const auto ok = plan(terms, out);
        record(
            "plan_slots_only_for_inactive_states",
            ok and out.original.size() == 2 and out.original[0].active_slot < 0
                and out.original[1].active_slot == 0
        );
    }
    {
        auto terms = eo_test::TermTableBuilder{};
        terms.add_reflection({0, 1}, {3}, {0.0, -0.7}, 0, 3);
        auto out = TermPlan{};
        const auto ok = plan(terms, out);
        const auto piece = first_piece(out.original);
        record(
            "plan_three_site_spectator_mask",
            ok and piece.local_state.site_count == 3 and piece.moving_count == 2
                and piece.bucket == Bucket::horizontal and piece.active_slot == 0
                and piece.is_one_hot
        );
    }
}
}

int main()
{
    run_validation_cases();
    run_plan_cases();
    std::printf(
        "[term_plan_test] summary status=%s failures=%d\n",
        g_failures == 0 ? "PASS" : "FAIL",
        g_failures
    );
    return g_failures == 0 ? 0 : 1;
}
