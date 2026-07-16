# cuQuantumNaturalfPEPS.jl
## Setup
If not done already clone the forked repo
```
git clone git@github.com:Daniel-Sinkin/QuantumNaturalfPEPS.jl.git
```

Start by activating the necessary modules
```
source ~/QuantumNaturalfPEPS.jl/cuQuantumNaturalfPEPS/util/load_modules.sh
```

You should see something like this after the per-module info:
```
[user@host QuantumNaturalfPEPS.jl]$ source ~/QuantumNaturalfPEPS.jl/cuQuantumNaturalfPEPS/util/load_modules.sh
cuQuantumNaturalfPEPS environment: julia version 1.12.3, Cuda compilation tools, release 13.0, V13.0.48
```

Set the location where Julia will put the dependencies, should put it in scratch space as the login folders have file count limits
```
export JULIA_DEPOT_PATH=/p/scratch/cslai/$USER/julia-peps-cuda
```

Jump into the repo
```
cd ~/QuantumNaturalfPEPS.jl
```

Download dependencies
```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=cuQuantumNaturalfPEPS -e 'using Pkg; Pkg.instantiate()'
```

Fresh rebuild
```
cd cuQuantumNaturalfPEPS
rm -rf cuda/build && ./setup_cuda.sh
```

An A100 devel clean build produced these identifying lines:
```
CUDA build architectures: 80
CUDA_VISIBLE_DEVICES: 0
NVIDIA GPUs (index, name, compute capability, driver):
0, NVIDIA A100-SXM4-40GB, 8.0, 595.71.05
1, NVIDIA A100-SXM4-40GB, 8.0, 595.71.05
2, NVIDIA A100-SXM4-40GB, 8.0, 595.71.05
3, NVIDIA A100-SXM4-40GB, 8.0, 595.71.05
-- The CXX compiler identification is GNU 14.3.0
-- The CUDA compiler identification is NVIDIA 13.0.48 with host compiler GNU 14.3.0
-- Found CUDAToolkit: /p/software/default/stages/2026/software/CUDA/13/targets/x86_64-linux/include;/p/software/default/stages/2026/software/CUDA/13/targets/x86_64-linux/include/cccl (found version "13.0.48")
[100%] Linking CUDA shared library libpeps_sampler.so
[100%] Built target peps_sampler
cuQuantumNaturalfPEPS 0.2 (2026-07-14)
```
That "cuQuantumNaturalfPEPS 0.2 (2026-07-14)" is the versioning for the compiled .so file, whenever the underlying code changes the library needs to be recompiled because you would otherwise use an outdated version of the library. The examples explicitly check for the version.

## Running the examples
You are now ready to run the examples, they should cover all functionality
```
julia --project=. app/load_peps.jl
julia --project=. app/double_layer.jl
julia --project=. app/double_layer_step.jl
julia --project=. app/sampling.jl
julia --project=. app/sampling_multigpu.jl
julia --project=. app/batched_rangefinder.jl
julia --project=. app/ffi.jl
```
