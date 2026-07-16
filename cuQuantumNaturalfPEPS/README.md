# cuQuantumNaturalfPEPS.jl

## Setup on JURECA

If not done already clone the forked repo
```
git clone git@github.com:Daniel-Sinkin/QuantumNaturalfPEPS.jl.git
```

The following command sets up the project and the CUDA dependencies.

```bash
source cuQuantumNaturalfPEPS/util/bootstrap_jureca.sh
```

By default the julia packaging information is placed at `$SCRATCH/$USER/julia-peps-cuda`. You can
set teh `QNPEPS_JULIA_DEPOT=...` environment variable beforehand to change that location.

You have to run this command in every new shell (terminal) to have all modules ready to use.
```bash
source cuQuantumNaturalfPEPS/util/load_modules.sh
```

Run this to re-compile the .so file of the library (80 here refers to the compute capability,
basically the Nvidia GPU generation, A100 has 80, H100/200 has 90).
```bash
cuQuantumNaturalfPEPS/setup_cuda.sh 80
```

It should print this version string
```text
cuQuantumNaturalfPEPS 0.3 (2026-07-14)
```
This string is checked to make sure that the compiled binary .so matches the program's requirements,
in general we'll just check that the versions are identical instead of supporting backward compatibility.

## Running the examples
The following are the examples which show off most of the functionality of this library so far.
```bash
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/load_peps.jl
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/double_layer.jl
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/double_layer_step.jl
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/sampling.jl
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/sampling_multigpu.jl
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/batched_rangefinder.jl
julia --project=cuQuantumNaturalfPEPS cuQuantumNaturalfPEPS/app/ffi.jl
```
