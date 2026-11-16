#include "test/eo/panel_capture.cuh"
#include "test/eo/panel_capture.h"

#include "core/error.hpp"
#include "linalg/transfer.cuh"

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <mutex>
#include <string_view>
#include <vector>

namespace qnpeps::test
{
namespace
{

constexpr std::string_view k_data_file{"panels.bin"};
constexpr std::string_view k_index_file{"panels_index.csv"};
constexpr std::string_view k_index_header{
    "record,event,evaluation,pass,first_sample,lane,sample,direction,depth,row,col,rows,cols,rank,"
    "device,panel_offset,basis_offset,factor_offset,sketch_offset\n"
};
constexpr i64 k_absent_offset{-1};
constexpr i64 k_no_evaluation{-1};

struct PanelCapture
{
    std::mutex lock{};
    bool armed{};
    bool in_pass{};
    PanelPass pass{};
    i64 first_sample{};
    i64 evaluation{k_no_evaluation};
    u64 event_stride{1};
    u64 event_limit{};
    usize lane_stride{1};
    usize lanes_per_event{};
    u64 truncation_events{};
    u64 captured_events{};
    u64 records{};
    u64 data_bytes{};
    std::ofstream data{};
    std::ofstream index{};
};

[[nodiscard]] auto panel_capture() -> PanelCapture&
{
    static PanelCapture capture{};
    return capture;
}

struct LaneOffsets
{
    i64 panel{};
    i64 basis{};
    i64 factor{};
    i64 sketch{};
};

[[nodiscard]] auto append_block(PanelCapture& capture, Linalg& linalg, const cf* source, i64 count)
    -> i64
{
    const auto element_count = static_cast<usize>(count);
    auto host = std::vector<cf>(element_count);
    download_async(linalg, host.data(), source, element_count);
    check(cudaStreamSynchronize(linalg.stream()));
    const auto byte_count = element_count * sizeof(cf);
    capture.data.write(
        reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(byte_count)
    );
    require(capture.data.good(), Status::internal, "[panel_capture] panel data write failed");
    const auto offset = static_cast<i64>(capture.data_bytes);
    capture.data_bytes += byte_count;
    return offset;
}

[[nodiscard]] auto pass_word(PanelPass pass) noexcept -> std::string_view
{
    switch (pass)
    {
        case PanelPass::original:
            return "original";
        case PanelPass::transposed:
            return "transposed";
    }
    return "unknown";
}

[[nodiscard]] auto _start_panel_capture_check(const QnpepsPanelCaptureSettings& settings) noexcept
    -> bool
{
    return settings.directory != nullptr and settings.event_stride >= 1
           and settings.lane_stride >= 1 and settings.lanes_per_event >= 1;
}

auto start_panel_capture(const QnpepsPanelCaptureSettings* settings) -> void
{
    require(settings != nullptr, Status::null_argument, "[panel_capture] settings are null");
    require_struct_size(*settings, "panel capture settings");
    require(
        _start_panel_capture_check(*settings),
        Status::bad_config,
        "[panel_capture] settings need a directory and strides and lane counts of at least one"
    );
    auto& capture = panel_capture();
    const std::scoped_lock guard{capture.lock};
    require(not capture.armed, Status::bad_config, "[panel_capture] a capture is already running");
    const std::filesystem::path directory{settings->directory};
    std::filesystem::create_directories(directory);
    capture.data = std::ofstream{directory / k_data_file, std::ios::binary | std::ios::trunc};
    capture.index = std::ofstream{directory / k_index_file, std::ios::trunc};
    require(
        capture.data.good() and capture.index.good(),
        Status::internal,
        "[panel_capture] cannot open the capture files in {}",
        directory.string()
    );
    capture.index << k_index_header;
    capture.armed = true;
    capture.in_pass = false;
    capture.evaluation = k_no_evaluation;
    capture.event_stride = settings->event_stride;
    capture.event_limit = settings->event_limit;
    capture.lane_stride = settings->lane_stride;
    capture.lanes_per_event = settings->lanes_per_event;
    capture.truncation_events = 0;
    capture.captured_events = 0;
    capture.records = 0;
    capture.data_bytes = 0;
}

auto finish_panel_capture(QnpepsPanelCaptureReading* out) -> void
{
    require(out != nullptr, Status::null_argument, "[panel_capture] reading output is null");
    require_struct_size(*out, "panel capture reading");
    auto& capture = panel_capture();
    const std::scoped_lock guard{capture.lock};
    out->truncation_events = capture.truncation_events;
    out->captured_events = capture.captured_events;
    out->records = capture.records;
    out->data_bytes = capture.data_bytes;
    if (not capture.armed) return;
    capture.data.close();
    capture.index.close();
    capture.armed = false;
    capture.in_pass = false;
    require(
        not capture.data.fail() and not capture.index.fail(),
        Status::internal,
        "[panel_capture] closing the capture files failed"
    );
}

}

auto enter_panel_pass(PanelPass pass, i64 first_sample) -> void
{
    auto& capture = panel_capture();
    const std::scoped_lock guard{capture.lock};
    if (not capture.armed) return;
    const auto starts_evaluation = pass == PanelPass::original and first_sample == 0;
    if (starts_evaluation) ++capture.evaluation;
    capture.in_pass = true;
    capture.pass = pass;
    capture.first_sample = first_sample;
}

auto leave_panel_pass() -> void
{
    auto& capture = panel_capture();
    const std::scoped_lock guard{capture.lock};
    capture.in_pass = false;
}

auto capture_panel(const PanelEvent& event) -> void
{
    auto& capture = panel_capture();
    const std::scoped_lock guard{capture.lock};
    if (not capture.armed or not capture.in_pass) return;
    const auto rows = event.panel.rows();
    const auto cols = event.panel.cols();
    const auto rank = event.basis.cols();
    if (rank >= std::min(rows, cols)) return;
    const auto candidate = capture.truncation_events++;
    if (candidate % capture.event_stride != 0) return;
    if (capture.captured_events >= capture.event_limit) return;
    const auto event_index = capture.captured_events++;
    int device{};
    check(cudaGetDevice(&device));
    auto& linalg = *event.linalg;
    check(cudaStreamSynchronize(linalg.stream()));
    const auto lane_count = static_cast<usize>(event.lanes);
    const auto depth = event.from_top ? event.row : event.lattice_rows - 1 - event.row;
    const std::string_view direction{event.from_top ? "from_top" : "from_bottom"};
    const auto panel_count = static_cast<i64>(rows) * cols;
    const auto basis_count = static_cast<i64>(rows) * rank;
    const auto factor_count = static_cast<i64>(rank) * cols;
    for (auto slot = 0_uz; slot < capture.lanes_per_event; ++slot)
    {
        const auto lane_step = slot * capture.lane_stride;
        if (lane_step >= lane_count) break;
        const auto lane = static_cast<i64>((event_index + lane_step) % lane_count);
        const auto offsets = LaneOffsets{
            .panel = append_block(
                capture, linalg, event.panel.data() + lane * event.panel.stride(), panel_count
            ),
            .basis = append_block(
                capture, linalg, event.basis.data() + lane * event.basis.stride(), basis_count
            ),
            .factor = append_block(
                capture, linalg, event.factor.data() + lane * event.factor.stride(), factor_count
            ),
            .sketch = event.sketch ? append_block(capture, linalg, event.sketch, factor_count)
                                   : k_absent_offset
        };
        capture.index << std::format(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            capture.records,
            event_index,
            capture.evaluation,
            pass_word(capture.pass),
            capture.first_sample,
            lane,
            capture.first_sample + lane,
            direction,
            depth,
            event.row,
            event.col,
            rows,
            cols,
            rank,
            device,
            offsets.panel,
            offsets.basis,
            offsets.factor,
            offsets.sketch
        );
        require(capture.index.good(), Status::internal, "[panel_capture] index write failed");
        ++capture.records;
    }
    capture.data.flush();
    capture.index.flush();
}

}

extern "C" qnpeps_status qnpeps_test_start_panel_capture(const QnpepsPanelCaptureSettings* settings)
{
    return qnpeps::guarded([&] { qnpeps::test::start_panel_capture(settings); });
}

extern "C" qnpeps_status qnpeps_test_finish_panel_capture(QnpepsPanelCaptureReading* out)
{
    return qnpeps::guarded([&] { qnpeps::test::finish_panel_capture(out); });
}
