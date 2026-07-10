#!/bin/bash
module load Stages/2026
module load GCC/14.3.0 ParaStationMPI/5.13.0-1 Julia/1.12.3
module load CUDA/13 cuTENSOR/2.3.1.0-CUDA-13
module load CMake
echo "cuQuantumNaturalfPEPS environment: $(julia --version 2>/dev/null || echo 'julia NOT on PATH'), $(nvcc --version 2>/dev/null | grep release || echo 'nvcc NOT on PATH')"
