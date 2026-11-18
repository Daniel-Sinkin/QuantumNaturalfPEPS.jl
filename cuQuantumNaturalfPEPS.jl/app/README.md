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
To pass your own operator matrices to the local energies and to a VMC step, run
```
julia --project=cuQuantumNaturalfPEPS.jl cuQuantumNaturalfPEPS.jl/app/custom_operators.jl
```
`convergence_l4_reference.jl` compares a run against stored reference data, which is not part of
this repository. `QNPEPS_CONVERGENCE_REFERENCE_ROOT` names the directory with the start state
under `fixtures/`, and `QNPEPS_CONVERGENCE_REFERENCE_WINDOW` names the directory with the
reference trajectory and its final states.
