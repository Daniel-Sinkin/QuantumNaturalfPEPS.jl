#ifndef QNPEPS_TEST_ORACLE_H
#define QNPEPS_TEST_ORACLE_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

struct OracleConfig
{
    int32_t lx;
    int32_t ly;
    int32_t dim_phys;
    int32_t dim_bond;
    int32_t chi_s;
    int32_t seed;
    int32_t dim_batch;
    int32_t batches;
    int32_t gpus;
    int32_t dl_host;
};

using OracleGeneratePeps = int(const OracleConfig*, float*);
using OracleCount = int64_t(const OracleConfig*);
using OracleBuildDlenv = int(const OracleConfig*, float*, int32_t*);
using OracleSample =
    int(const OracleConfig*, const float*, const void*, const int32_t*, uint64_t*, double*);

template <typename Function>
[[nodiscard]] inline auto oracle_symbol(void* handle, const char* name) -> Function*
{
    return reinterpret_cast<Function*>(dlsym(handle, name));
}

[[nodiscard]] inline auto oracle_so_path() -> const char*
{
    const auto* from_env = std::getenv("QNPEPS_ORACLE_SO");
    if (not from_env)
    {
        std::fprintf(
            stderr,
            "QNPEPS_ORACLE_SO is not set; point it at the frozen src/ oracle .so "
            "(the repo-root build-cuda/lib/libpeps_sampler.so)\n"
        );
        std::exit(1);
    }
    return from_env;
}

#endif
