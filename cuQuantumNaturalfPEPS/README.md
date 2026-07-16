# cuQuantumNaturalfPEPS.jl
## Setup
The following command sets up the project and the CUDA dependencies. This has to be done once and can take a bit of time.
```bash
source cuQuantumNaturalfPEPS/setup.sh
```

You should see something akin to this:
```
cuQuantumNaturalfPEPS JURECA toolchain
  Julia: julia version 1.12.3
  CUDA compiler: Build cuda_13.0.r13.0/compiler.36260728_0
  CUDA_HOME: /p/software/default/stages/2026/software/CUDA/13
  JULIA_DEPOT_PATH: /p/scratch/cslai/sinkin1/julia-peps-cuda
...
  Activating project at `~/QuantumNaturalfPEPS.jl`
  Activating project at `~/QuantumNaturalfPEPS.jl/cuQuantumNaturalfPEPS`
...
Building libpeps_sampler.so
CUDA build architectures: 80
CUDA_VISIBLE_DEVICES: 0,1,2,3
NVIDIA GPUs (index, name, compute capability, driver):
0, NVIDIA A100-SXM4-40GB, 8.0, 595.71.05
1, NVIDIA A100-SXM4-40GB, 8.0, 595.71.05
2, NVIDIA A100-SXM4-40GB, 8.0, 595.71.05
3, NVIDIA A100-SXM4-40GB, 8.0, 595.71.05
...
```

By default the julia packaging information is placed at `$SCRATCH/$USER/julia-peps-cuda`. You can
set the `QNPEPS_JULIA_DEPOT=...` environment variable beforehand to change that location.

Once that is done every new shell (terminal) needs to run this once to activate the modules and dependencies
```bash
source cuQuantumNaturalfPEPS/activate.sh
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
chi_dl = dim_bond^2 = 16 is exact; default chi_dl = dim_bond truncates.
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
