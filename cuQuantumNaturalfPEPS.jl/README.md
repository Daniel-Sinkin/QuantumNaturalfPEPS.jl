This library gets build and compiled with a single shell script, in the root of the repository just run
```
source cuQuantumNaturalfPEPS.jl/setup.sh
```
This builds the library to a fixed location, it is possible to change this location but I will document that later, should not be necessary at this stage.

After this is done, take a look at `app/`, there you will find examples of how to call the functions, start with reading the `README.md` (in general should always be done first when looking at a folder), currently quite sparse but will be extended as the documentation matures.

## Tests

The Julia test suite is `test/runtests.jl`. The tier 2 minSR replay testset is skipped unless `QNPEPS_MINSR_REPLAY_DIRS` is set to a comma separated list of fixture directories. When it is set, `QNPEPS_MINSR_REPLAY_OUT` must name the directory that receives the replay output, and the testset fails if it is empty.
