#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "GPUTimer.h"



GPUTimer::GPUTimer() 
{
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_, 0);
}

GPUTimer::~GPUTimer() 
{
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

void GPUTimer::start() 
{
    cudaEventRecord(start_, 0);
}

float GPUTimer::seconds() 
{
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
    float time;
    cudaEventElapsedTime(&time, start_, stop_);
    return time * 1e-3;
}