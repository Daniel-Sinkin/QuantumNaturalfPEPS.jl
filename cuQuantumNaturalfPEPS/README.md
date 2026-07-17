# cuQuantumNaturalfPEPS.jl
## Setup
The following command sets up the project and the CUDA dependencies. This has to be done once and can take a bit of time.
```bash
source cuQuantumNaturalfPEPS/setup.sh
```

You should see something akin to this (Visible NVIDIA GPUs of course is different on Login and Compute Nodes):
```
cuQuantumNaturalfPEPS JURECA toolchain
  Host: jrlogin04.jureca
  Julia: julia version 1.12.3
  Julia executable: /p/software/fs/jurecadc/stages/2026/software/Julia/1.12.3-gpsmpi-2025b/bin/julia
  CUDA compiler: Build cuda_13.0.r13.0/compiler.36260728_0
  CUDA compiler executable: /p/software/fs/jurecadc/stages/2026/software/CUDA/13/bin/nvcc
  CUDA_HOME: /p/software/default/stages/2026/software/CUDA/13
  CUDA toolkit libraries: /p/software/fs/jurecadc/stages/2026/software/CUDA/13/targets/x86_64-linux/lib
  JULIA_DEPOT_PATH: /p/scratch/cslai/sinkin1/julia-peps-cuda
  QNPEPS_ACTIVE_ROOT: /p/home/jusers/sinkin1/jureca/QuantumNaturalfPEPS.jl/cuQuantumNaturalfPEPS
  PATH: ...
  LD_LIBRARY_PATH: ...
  LD_PRELOAD: <unset>
  Visible NVIDIA GPUs (index, name, compute capability, driver):
    0, Quadro RTX 8000, 7.5, 595.71.05
    1, Quadro RTX 8000, 7.5, 595.71.05
...
  Activating project at `~/QuantumNaturalfPEPS.jl`
  Activating project at `~/QuantumNaturalfPEPS.jl/cuQuantumNaturalfPEPS`
...
Building qnpeps.so
CMake: cmake version 4.0.3
CMake executable: /p/software/fs/jurecadc/stages/2026/software/CMake/4.0.3-GCCcore-14.3.0/bin/cmake
CUDA build architectures: 75;80;90
CUDA_VISIBLE_DEVICES: <unset>
NVIDIA GPUs (index, name, compute capability, driver):
0, Quadro RTX 8000, 7.5, 595.71.05
1, Quadro RTX 8000, 7.5, 595.71.05
...
cuQuantumNaturalfPEPS 0.0.4 (2026-07-17)
Native CUDA images: cubin=sm_75,sm_80,sm_90 ptx=sm_75,sm_80,sm_90
Resolved CUDA library dependencies:
  libcudart.so.13 => .../libcudart.so.13.0.48
  libcublas.so.13 => .../libcublas.so.13.0.0.19
  libcusolver.so.12 => .../libcusolver.so.12.0.3.29
  libcurand.so.10 => .../libcurand.so.10.4.0.35
  libcublasLt.so.13 => .../libcublasLt.so.13.0.0.19
  libcusparse.so.12 => .../libcusparse.so.12.6.2.49
  libnvJitLink.so.13 => .../libnvJitLink.so.13.0.39
Inspecting Julia and native CUDA runtime
cuQuantumNaturalfPEPS runtime
  native library: .../cuda/build/qnpeps.so
  loaded C API: cuQuantumNaturalfPEPS 0.0.4 (2026-07-17)
CUDA toolchain:
- runtime 13.0, local installation
- driver 595.71.5 for 13.3
- compiler 13.3, artifact installation
CUDA libraries:
- CUBLAS: 13.0.0
- CUSOLVER: 12.0.3
- CUSPARSE: 12.6.2
...
Loaded native library paths:
- .../CUDA/13/lib/libcublas.so => .../libcublas.so.13.0.0.19
- .../CUDA/13/lib/libcusolver.so => .../libcusolver.so.12.0.3.29
- .../Julia/1.12.3-gpsmpi-2025b/lib/julia/libcurl.so.4 => .../libcurl.so.4.8.0
```

By default the julia packaging information is placed at `$SCRATCH/$USER/julia-peps-cuda`. You can
set the `QNPEPS_JULIA_DEPOT=...` environment variable beforehand to change that location.

The library always contains native CUDA code for compute capabilities 7.5, 8.0, and 9.0.

Once that is done every new shell (terminal) needs to run this once to activate the modules and dependencies
```bash
source cuQuantumNaturalfPEPS/activate.sh
```

Successful activation sets `QNPEPS_ACTIVE_ROOT` to the package directory.
```bash
test "${QNPEPS_ACTIVE_ROOT:-}" = "$(cd cuQuantumNaturalfPEPS && pwd)"
```

## Running the examples
The following are the examples which show off most of the functionality of this library so far.
```bash
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/load_peps.jl
```
You should see this output
```bash
arrays_grid (4, 4) dim_bond 4
itensor_grid (4, 4) dim_bond 4
corner_dims (2, 1, 4, 4, 1)
center_dims (2, 4, 4, 4, 4)
device_buffer_elements 3200 device CuDevice(0)
```

```bash
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/double_layer.jl
```
You should see this output
```bash
peps: 4 x 4 lattice, dim_bond 4, dim_phys 2
chi_dl caps the bond dimension of the double-layer row MPS.
chi_dl = dim_bond^2 = 16 is exact, while default chi_dl = dim_bond truncates.
untruncated: chi_dl 16, env bytes 393408
truncated: chi_dl 4, env bytes 24768
cumulative row log-norms untruncated: [-3.9403818791584007, -3.0956433956063267, -1.0088134003043083]
cumulative row log-norms truncated: [-4.507111354172459, -3.5654054358990415, -2.0095780338371423]
max row-log deviation 1.000764633532834 at 6.3% of the env memory
```

```bash
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/double_layer_step.jl
```

You should see this output
```bash
row 3 sites 4 maxlinkdim 16 row_log -1.0088134003043083
row 2 sites 4 maxlinkdim 16 row_log -1.9520213827444928
row 1 sites 4 maxlinkdim 16 row_log -0.8518327714826055
step_cumulative_row_logs [-3.812667554531407, -2.960834783048801, -1.0088134003043083]
oneshot_cumulative_row_logs [-3.9403818791584007, -3.0956433956063267, -1.0088134003043083]
max_abs_diff 0.13480861255752563
```

```bash
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/sampling.jl
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/sampling_multigpu.jl
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/batched_rangefinder.jl
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/ffi.jl
```
