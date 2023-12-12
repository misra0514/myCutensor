#include <stdlib.h>
#include <stdio.h>

#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>
#include <cuComplex.h>
#include <curand_kernel.h>

#include "kernel.h"



__global__ void init_data_gpu(cuFloatComplex* arr, uint64_t n) {
    uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        arr[i].x = (float)(i%512)+1.111f;
        arr[i].y = (float)(i%512)+1.111f;
    }
    
    if (i < 5)
        printf("** index: %llu ** gpu: (%f, %f)\n", i, float(arr[i].x), float(arr[i].y));
}

__global__ void init_data_unified(cuFloatComplex* arr, uint64_t n) {
    uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        arr[i].x = (float)(i%512)+1.111f;
        arr[i].y = (float)(i%512)+1.111f;
    }
    
    if (i < 5)
        printf("** index: %llu ** unified: (%f, %f)\n", i, float(arr[i].x), float(arr[i].y));
}

__global__ void check_out_kernel(cuFloatComplex* C_gpu, cuFloatComplex* C_d, uint64_t n) {
    uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        if ( (abs(float(C_gpu[i].x-C_d[i].x))>=1e-7 || abs(float(C_gpu[i].y-C_d[i].y))>=1e-7) ) {
            //printf("** index: %llu ** Goes Wrong (%f, %f)\n", i, float(C_gpu[i].x-C_d[i].x), float(C_gpu[i].y-C_d[i].y));
        }
    }
    if (i < 5)
        printf("gpu: (%f, %f), unified: (%f, %f)\n", float(C_gpu[i].x), float(C_gpu[i].y), float(C_d[i].x), float(C_d[i].y));
}