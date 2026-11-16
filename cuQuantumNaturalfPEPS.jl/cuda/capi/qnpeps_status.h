#ifndef QNPEPS_STATUS_H
#define QNPEPS_STATUS_H

#ifdef __cplusplus
extern "C"
{
#    pragma GCC visibility push(default)
#endif

    typedef enum qnpeps_status
    {
        QNPEPS_OK = 0,
        QNPEPS_ERR_NULL_ARG = 1,
        QNPEPS_ERR_BAD_CONFIG = 2,
        QNPEPS_ERR_BAD_VERSION = 3,
        QNPEPS_ERR_CUDA = 4,
        QNPEPS_ERR_OOM = 5,
        QNPEPS_ERR_INTERNAL = 6
    } qnpeps_status;

    const char* qnpeps_last_error(void);

#ifdef __cplusplus
#    pragma GCC visibility pop
}
#endif

#endif
