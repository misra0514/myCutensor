#include <stdlib.h>
#include <stdio.h>

#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>
#include <cuComplex.h>
#include <curand_kernel.h>

#include "kernel.h"
#include "Val.h"
#include "Contraction.h"
#include "GPUTimer.h"

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("Error: %s\n", cudaGetErrorString(err)); return err; } \
};

typedef cuFloatComplex floatTypeA;
typedef cuFloatComplex floatTypeB;
typedef cuFloatComplex floatTypeC;
typedef cuFloatComplex floatTypeCompute;


bool multiBatch::cudaMallocManaged_init(void** A_d_t, void** B_d_t, void** C_d_t, 
                                        uint64_t sizeA, uint64_t sizeB, uint64_t sizeC, 
                                        uint64_t elementsA, uint64_t elementsB, uint64_t elementsC) {
    HANDLE_CUDA_ERROR(cudaMallocManaged(A_d_t, sizeA));
    HANDLE_CUDA_ERROR(cudaMallocManaged(B_d_t, sizeB));
    HANDLE_CUDA_ERROR(cudaMallocManaged(C_d_t, sizeC));
    cudaDeviceSynchronize();
    
    init_data_unified<<<(elementsA+255)/256, 256>>>((floatTypeA*)*A_d_t, elementsA);
    init_data_unified<<<(elementsB+255)/256, 256>>>((floatTypeB*)*B_d_t, elementsB);
    init_data_unified<<<(elementsC+255)/256, 256>>>((floatTypeC*)*C_d_t, elementsC);
    cudaDeviceSynchronize();

    printf("Malloced Unified Memory: %.2f GiB\n", (sizeA + sizeB + sizeC)/1024./1024./1024);
    fflush(stdout);
    
    return true;    
}

bool multiBatch::cudaMallocManaged_ABC(void** A_d_t, void** B_d_t, void** C_d_t, 
                                        uint64_t sizeA, uint64_t sizeB, uint64_t sizeC) {
    HANDLE_CUDA_ERROR(cudaMallocManaged(A_d_t, sizeA));
    HANDLE_CUDA_ERROR(cudaMallocManaged(B_d_t, sizeB));
    HANDLE_CUDA_ERROR(cudaMallocManaged(C_d_t, sizeC));
    cudaDeviceSynchronize();
    
    printf("Malloced Unified Memory: %.2f GiB\n", (sizeA + sizeB + sizeC)/1024./1024./1024);
    fflush(stdout);
    
    return true;
}

void multiBatch::init_unified_ABC(void* A_d, void* B_d, void* C_d, 
                      uint64_t elementsA, uint64_t elementsB, uint64_t elementsC) {
    init_data_unified<<<(elementsA+255)/256, 256>>>((floatTypeA*)A_d, elementsA);
    init_data_unified<<<(elementsB+255)/256, 256>>>((floatTypeB*)B_d, elementsB);
    init_data_unified<<<(elementsC+255)/256, 256>>>((floatTypeC*)C_d, elementsC);
    cudaDeviceSynchronize();
}

bool multiBatch::cudaMalloc_init(void** A_gpu_t, void** B_gpu_t, void** C_gpu_t, 
                                uint64_t sizeA, uint64_t sizeB, uint64_t sizeC, 
                                uint64_t elementsA, uint64_t elementsB, uint64_t elementsC) {
    HANDLE_CUDA_ERROR(cudaMalloc(A_gpu_t, sizeA));
    HANDLE_CUDA_ERROR(cudaMalloc(B_gpu_t, sizeB));
    HANDLE_CUDA_ERROR(cudaMalloc(C_gpu_t, sizeC));
    cudaDeviceSynchronize();
    
    init_data_gpu<<<(elementsA+255)/256, 256>>>((floatTypeA*)*A_gpu_t, elementsA);
    init_data_gpu<<<(elementsB+255)/256, 256>>>((floatTypeB*)*B_gpu_t, elementsB);
    init_data_gpu<<<(elementsC+255)/256, 256>>>((floatTypeC*)*C_gpu_t, elementsC);
    cudaDeviceSynchronize();
    
    printf("Malloced Global Memory: %.2f GiB\n", (sizeA + sizeB + sizeC)/1024./1024./1024);
    fflush(stdout);
    
    return true;
}

void multiBatch::cal_unified(void *A_d, void *B_d, void *C_d, 
                    cudaDataType_t typeA, cudaDataType_t typeB, cudaDataType_t typeC, cutensorComputeType_t typeCompute,
                    int nmodeA, int nmodeB, int nmodeC,
                    std::vector<int> modeA, std::vector<int> modeB, std::vector<int> modeC,
                    std::vector<int64_t> extentA, std::vector<int64_t> extentB, std::vector<int64_t> extentC,
                    int algo) {

    /* contraction */
    Contraction contraction = Contraction(A_d, B_d, C_d, make_cuFloatComplex(1.0f, 0.f), make_cuFloatComplex(0.f, 0.f));
    contraction.init(typeA, typeB, typeC, typeCompute,
                nmodeA, nmodeB, nmodeC, 
                modeA, modeB, modeC,
                extentA, extentB, extentC,
                algo);
    
    double minTimeCUTENSOR = 1e100;
    for (int i = 0; i < 3; ++i) {
         // Set up timing
        GPUTimer timer;
        timer.start();

        contraction.execute();

        // Synchronize and measure timing
        auto time = timer.seconds();
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
        printf("(unified)loop %d: %f ms\n", i, time*1000);
    }
    printf("(unified)best: %f ms\n", minTimeCUTENSOR*1000);
    
    cudaDeviceSynchronize();
}

void multiBatch::cal_gpu(void *A_gpu, void *B_gpu, void *C_gpu, 
        cudaDataType_t typeA, cudaDataType_t typeB, cudaDataType_t typeC, cutensorComputeType_t typeCompute,
        int nmodeA, int nmodeB, int nmodeC,
        std::vector<int> modeA, std::vector<int> modeB, std::vector<int> modeC,
        std::vector<int64_t> extentA, std::vector<int64_t> extentB, std::vector<int64_t> extentC,
        int algo) {

    /* contraction */
    Contraction contraction = Contraction(A_gpu, B_gpu, C_gpu, make_cuFloatComplex(1.0f, 0.f), make_cuFloatComplex(0.f, 0.f));
    contraction.init(typeA, typeB, typeC, typeCompute,
                nmodeA, nmodeB, nmodeC, 
                modeA, modeB, modeC,
                extentA, extentB, extentC,
                algo);
    
    double minTimeCUTENSOR = 1e100;
    for (int i = 0; i < 3; ++i) {
         // Set up timing
        GPUTimer timer;
        timer.start();

        contraction.execute();

        // Synchronize and measure timing
        auto time = timer.seconds();
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
        printf("(gpu)loop %d: %f ms\n", i, time*1000);
    }
    printf("(gpu)best: %f ms\n", minTimeCUTENSOR*1000);
    
    cudaDeviceSynchronize();
}

void multiBatch::check_out(void *C_gpu, void *C_d, uint64_t info_unified_elementsC, uint64_t info_gpu_elementsC) {
    printf("  ********** Start check out the output **********\n");
    printf("  ******* info_unified_elementsC: %lld *******\n", info_unified_elementsC);
    printf("  ******* info_unified_elementsC: %lld *******\n\n", info_gpu_elementsC);
    printf("C_gpu: %p\n", C_gpu);
    printf("C_d: %p\n", C_d);
    
    for (int i = 0; i < (info_unified_elementsC/info_gpu_elementsC); i++) {
        printf("  ********** %lld Check **********\n", i);
        check_out_kernel<<<(info_gpu_elementsC+255)/256, 256>>>((floatTypeC*)C_gpu, &((floatTypeC*)C_d)[i*info_gpu_elementsC], info_gpu_elementsC);
        cudaDeviceSynchronize();
        printf("  ********** %lld Check Finish **********\n", i);
    }

    //check_out_kernel<<<(info_gpu_elementsC+255)/256, 256>>>((floatTypeC*)C_gpu, (floatTypeC*)C_d, info_gpu_elementsC);
    
    cudaDeviceSynchronize();
    printf("\n  ********** finish check out **********\n");
    
}


Val::Val() {}

bool Val::cudaMallocManaged_init(uint64_t sizeA, uint64_t sizeB, uint64_t sizeC, 
                                uint64_t elementsA, uint64_t elementsB, uint64_t elementsC) {
    HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &A_d_, sizeA));
    HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &B_d_, sizeB));
    HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &C_d_, sizeC));
    cudaDeviceSynchronize();
    
    init_data_unified<<<(elementsA+255)/256, 256>>>((floatTypeA*)A_d_, elementsA);
    init_data_unified<<<(elementsB+255)/256, 256>>>((floatTypeB*)B_d_, elementsB);
    init_data_unified<<<(elementsC+255)/256, 256>>>((floatTypeC*)C_d_, elementsC);
    cudaDeviceSynchronize();

    printf("Malloced Unified Memory: %.2f GiB\n", (sizeA + sizeB + sizeC)/1024./1024./1024);
    fflush(stdout);
    
    return true;
}

bool Val::cudaMalloc_init(uint64_t sizeA, uint64_t sizeB, uint64_t sizeC, 
                        uint64_t elementsA, uint64_t elementsB, uint64_t elementsC) {
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &A_gpu_, sizeA));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &B_gpu_, sizeB));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &C_gpu_, sizeC));
    cudaDeviceSynchronize();
    
    init_data_gpu<<<(elementsA+255)/256, 256>>>((floatTypeA*)A_gpu_, elementsA);
    init_data_gpu<<<(elementsB+255)/256, 256>>>((floatTypeB*)B_gpu_, elementsB);
    init_data_gpu<<<(elementsC+255)/256, 256>>>((floatTypeC*)C_gpu_, elementsC);
    cudaDeviceSynchronize();
    
    printf("Malloced Global Memory: %.2f GiB\n", (sizeA + sizeB + sizeC)/1024./1024./1024);
    fflush(stdout);
    
    return true;
}

void Val::cal_unified(cudaDataType_t typeA, cudaDataType_t typeB, cudaDataType_t typeC, cutensorComputeType_t typeCompute,
        int nmodeA, int nmodeB, int nmodeC,
        std::vector<int> modeA, std::vector<int> modeB, std::vector<int> modeC,
        std::vector<int64_t> extentA, std::vector<int64_t> extentB, std::vector<int64_t> extentC,
        int algo) {

    /* contraction */
    Contraction contraction = Contraction(A_d_, B_d_, C_d_, make_cuFloatComplex(1.0f, 0.f), make_cuFloatComplex(0.f, 0.f));
    contraction.init(typeA, typeB, typeC, typeCompute,
                nmodeA, nmodeB, nmodeC, 
                modeA, modeB, modeC,
                extentA, extentB, extentC,
                algo);
    
    double minTimeCUTENSOR = 1e100;
    for (int i = 0; i < 3; ++i) {
         // Set up timing
        GPUTimer timer;
        timer.start();

        contraction.execute();

        // Synchronize and measure timing
        auto time = timer.seconds();
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
        printf("(unified)loop %d: %f ms\n", i, time*1000);
    }
    printf("(unified)best: %f ms\n", minTimeCUTENSOR*1000);
    
    cudaDeviceSynchronize();
}

void Val::cal_gpu(cudaDataType_t typeA, cudaDataType_t typeB, cudaDataType_t typeC, cutensorComputeType_t typeCompute,
        int nmodeA, int nmodeB, int nmodeC,
        std::vector<int> modeA, std::vector<int> modeB, std::vector<int> modeC,
        std::vector<int64_t> extentA, std::vector<int64_t> extentB, std::vector<int64_t> extentC,
        int algo) {

    /* contraction */
    Contraction contraction = Contraction(A_gpu_, B_gpu_, C_gpu_, make_cuFloatComplex(1.0f, 0.f), make_cuFloatComplex(0.f, 0.f));
    contraction.init(typeA, typeB, typeC, typeCompute,
                nmodeA, nmodeB, nmodeC, 
                modeA, modeB, modeC,
                extentA, extentB, extentC,
                algo);
    
    double minTimeCUTENSOR = 1e100;
    for (int i = 0; i < 3; ++i) {
         // Set up timing
        GPUTimer timer;
        timer.start();

        contraction.execute();

        // Synchronize and measure timing
        auto time = timer.seconds();
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
        printf("(gpu)loop %d: %f ms\n", i, time*1000);
    }
    printf("(gpu)best: %f ms\n", minTimeCUTENSOR*1000);
    
    cudaDeviceSynchronize();
}

void Val::check_out(uint64_t info_unified_elementsC, uint64_t info_gpu_elementsC) {
    printf("  ********** Start check out the output **********\n");
    printf("  ******* info_unified_elementsC: %lld *******\n", info_unified_elementsC);
    printf("  ******* info_unified_elementsC: %lld *******\n\n", info_gpu_elementsC);
    printf("C_gpu: %p\n", C_gpu_);
    printf("C_d: %p\n", C_d_);
    
    for (int i = 0; i < (info_unified_elementsC/info_gpu_elementsC); i++) {
        printf("  ********** %lld Check **********\n", i);
        check_out_kernel<<<(info_gpu_elementsC+255)/256, 256>>>((floatTypeC*)C_gpu_, &((floatTypeC*)C_d_)[i*info_gpu_elementsC], info_gpu_elementsC);
        cudaDeviceSynchronize();
        printf("  ********** %lld Check Finish **********\n", i);
    }

    //check_out_kernel<<<(info_gpu_elementsC+255)/256, 256>>>((floatTypeC*)C_gpu_, (floatTypeC*)C_d_, info_gpu_elementsC);
    
    cudaDeviceSynchronize();
    printf("\n  ********** finish check out **********\n");
    
}


Val::~Val() {
    if (A_d_) cudaFree(A_d_);
    if (B_d_) cudaFree(B_d_);
    if (C_d_) cudaFree(C_d_);
    if (A_gpu_) cudaFree(A_gpu_);
    if (B_gpu_) cudaFree(B_gpu_);
    if (C_gpu_) cudaFree(C_gpu_);
}
