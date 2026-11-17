// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
// The step kernel of mmgp.graph_loop, the last kernel of a loop body: counter += 1, and the WHILE node runs another turn while
// counter < iterations. cudaGraphSetConditional comes from the CUDA device runtime, which nvcc resolves into machine code: the kernel
// ships as a fatbin of SASS (PTX alone would leave it unresolved). Built with CUDA 12.8 (MSVC on Windows):
//   nvcc -fatbin -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86
//        -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100
//        -gencode=arch=compute_120,code=sm_120 graph_loop_step.cu -o graph_loop_step.fatbin
#include <cuda_runtime.h>

extern "C" __global__ void mmgp_loop_step(long long* counter, long long iterations, cudaGraphConditionalHandle handle) {
    const long long next = counter[0] + 1;
    counter[0] = next;
    cudaGraphSetConditional(handle, next < iterations ? 1 : 0);
}
