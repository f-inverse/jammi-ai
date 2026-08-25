#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define C10_CUDA_CHECK(EXPR) do { cudaError_t __e = (EXPR); if (__e != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(__e), __FILE__, __LINE__); abort(); } } while (0)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
