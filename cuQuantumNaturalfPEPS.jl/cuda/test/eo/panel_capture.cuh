#ifndef QNPEPS_TEST_PANEL_CAPTURE_CUH
#define QNPEPS_TEST_PANEL_CAPTURE_CUH

#include "linalg/eo.cuh"

namespace qnpeps::test
{

enum class PanelPass : i32
{
    original = 0,
    transposed = 1
};

struct PanelEvent
{
    Linalg* linalg{};
    int row{};
    int col{};
    bool from_top{};
    int lattice_rows{};
    int lanes{};
    const cf* sketch{};
    MatrixBatch panel;
    MatrixBatch basis;
    MatrixBatch factor;
};

auto enter_panel_pass(PanelPass pass, i64 first_sample) -> void;
auto leave_panel_pass() -> void;
auto capture_panel(const PanelEvent& event) -> void;

}

#endif
