# pragma once

__global__ void init_data_gpu(cuFloatComplex* arr, uint64_t n);

__global__ void init_data_unified(cuFloatComplex* arr, uint64_t n);

__global__ void check_out_kernel(cuFloatComplex* C_gpu, cuFloatComplex* C_d, uint64_t n);