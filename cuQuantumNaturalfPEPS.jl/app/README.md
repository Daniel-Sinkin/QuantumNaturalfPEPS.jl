Make sure that you have compiled the library by first running
```sh
cuQuantumNaturalfPEPS.jl/util/build_cuda.sh
export QNPEPS_LIB="$PWD/cuQuantumNaturalfPEPS.jl/build/cuda/qnpeps.so"
```
you can then run 
```
`julia --project=cuQuantumNaturalfPEPS.jl cuQuantumNaturalfPEPS.jl/app/X.jl
```
where X is some filename. Best place to start would be 
```
`julia --project=cuQuantumNaturalfPEPS.jl cuQuantumNaturalfPEPS.jl/app/basic_usage.jl
```
