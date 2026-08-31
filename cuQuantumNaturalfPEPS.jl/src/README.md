This holds all of the Julia Code. Currently this is (mostly) a wrapper around the CUDA endpoints with some syntactic sugar, goal would be to move more of the CUDA code into Julia (especially things like non-hotpath bookkeeping).

This library currently exports way too many symbols.
