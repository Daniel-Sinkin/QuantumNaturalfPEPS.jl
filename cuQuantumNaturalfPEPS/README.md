# cuQuantumNaturalfPEPS.jl

## Setup on JURECA

If needed, clone the fork:

```
git clone git@github.com:Daniel-Sinkin/QuantumNaturalfPEPS.jl.git
```

On JURECA, the `CUDA/13` module is the toolkit-provisioning mechanism: it makes `nvcc`, headers,
and runtime libraries available without requiring root access or a separate CUDA download.

Run the fresh Julia instantiation and precompilation on a CPU compute node. From the repository
root, one sourced command configures the module toolchain, places the Julia depot under project
scratch, pins CUDA.jl to the module-provided CUDA 13 runtime, instantiates both Julia projects, and
builds the A100 library:

```bash
source cuQuantumNaturalfPEPS/util/bootstrap_jureca.sh
```

The default depot is `$SCRATCH/$USER/julia-peps-cuda`. Set
`QNPEPS_JULIA_DEPOT=/path/on/scratch` before sourcing the bootstrap to override it.

In each later shell, restore the runtime environment before running Julia:

```bash
source cuQuantumNaturalfPEPS/util/load_modules.sh
```

To rebuild only the native library after the environment is loaded:

```bash
cuQuantumNaturalfPEPS/setup_cuda.sh 80
```

The build prints the selected architecture, visible GPUs when available, compiler output, library
path, and ABI version. This branch must report:

```text
cuQuantumNaturalfPEPS 0.3 (2026-07-14)
```

Julia checks that ABI string when it loads the library. Rebuild the shared library after changing
the CUDA sources or ABI.

## Running the examples

From the repository root, run:

```bash
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/load_peps.jl
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/double_layer.jl
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/double_layer_step.jl
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/sampling.jl
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/sampling_multigpu.jl
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/batched_rangefinder.jl
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/ffi.jl
```
