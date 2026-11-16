#ifndef QNPEPS_TEST_PANEL_CAPTURE_H
#define QNPEPS_TEST_PANEL_CAPTURE_H

#include "qnpeps_status.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#    pragma GCC visibility push(default)
#endif

    typedef struct QnpepsPanelCaptureSettings
    {
        uint32_t struct_size;
        uint32_t event_stride;
        uint32_t event_limit;
        uint32_t lane_stride;
        uint32_t lanes_per_event;
        uint32_t reserved;
        const char* directory;
    } QnpepsPanelCaptureSettings;

    typedef struct QnpepsPanelCaptureReading
    {
        uint32_t struct_size;
        uint32_t reserved;
        uint64_t truncation_events;
        uint64_t captured_events;
        uint64_t records;
        uint64_t data_bytes;
    } QnpepsPanelCaptureReading;

    qnpeps_status qnpeps_test_start_panel_capture(const QnpepsPanelCaptureSettings* settings);
    qnpeps_status qnpeps_test_finish_panel_capture(QnpepsPanelCaptureReading* out);

#ifdef __cplusplus
#    pragma GCC visibility pop
}
#endif

#endif
