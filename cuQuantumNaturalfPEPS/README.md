# cuQuantumNaturalfPEPS.jl

## Setup and activation

To use the library on the cluster you need to compile the newest version and activate the required modules.

This is done via the same script, which detects if there is an up to date compiled version or not. Note that
this fused the previously used activation.sh and setup.sh into one script, so you just have use it independent
of if the library is compiled or not.

The compiled .so library will work on compute capabilities:
 * 7.5 which is JURECA login node
 * 8.0 which is the A100 generation (JURECA compute)
 * 9.0 which is the H100/200 generation (Jupiter compute).

This library was never tested on blackwell cards (B200, compute capability 10.0)

Make sure your are in the repo root and simply run this command. 
It will detect whether you need to recompile or just active the modules.
```bash
source cuQuantumNaturalfPEPS/setup.sh
```
If you want to force a recompilation you can use
```bash
source cuQuantumNaturalfPEPS/setup.sh --force
```
By default the $PATH is not printed as it is a lot of text, if you want it to show then use
```bash
source cuQuantumNaturalfPEPS/setup.sh --show-path
```
If you run into any compilation issues please re-run it with --show-path and provide the entire terminal output
in the corresponding Github issue.

Julia tends to use a lot of space and create a lot of small files. This can cause problems on the cluster
as the there are storage and filecount limits, so by default the temporary files (packages, precompilation data and so on)
are written on  the users scratch space at `$SCRATCH/$USER/julia-peps-cuda` . If you want to change this then simply set
the `QNPEPS_JULIA_DEPOT` environment variable to that filepath before running the setup.

Note that you must use `source` to activate the modules because those are only valid for that shell, in particular
you must use the library in the same shell that you activated the library in. I don't think there is a way around 
this with the way that the module loading is handled on the JSC cluster.

If the activation is successful, then the `QNPEPS_ACTIVE_ROOT` environment variable is set. You can check if the
environment for this checkout is active by running this command.
```bash
source cuQuantumNaturalfPEPS/setup.sh --check
```

You might see a warning that cuDNN is not compiled or not found, that is safe to ignore.

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
