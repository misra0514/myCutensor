/*  
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 * 
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name(s) of the copyright holder(s) nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */  

#include <stdlib.h>
#include <stdio.h>

#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>
#include <cuComplex.h>


#define HANDLE_ERROR(x)                                               \
{ const auto err = x;                                                 \
  if( err != CUTENSOR_STATUS_SUCCESS )                                \
  { printf("Error: %s\n", cutensorGetErrorString(err)); return err; } \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("Error: %s\n", cudaGetErrorString(err)); return err; } \
};


typedef cuFloatComplex floatTypeA;
typedef cuFloatComplex floatTypeB;
typedef cuFloatComplex floatTypeC;
typedef cuFloatComplex floatTypeCompute;

cudaDataType_t typeA = CUDA_C_32F;
cudaDataType_t typeB = CUDA_C_32F;
cudaDataType_t typeC = CUDA_C_32F;
cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_TF32;


struct GPUTimer
{
    GPUTimer() 
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, 0);
    }

    ~GPUTimer() 
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() 
    {
        cudaEventRecord(start_, 0);
    }

    float seconds() 
    {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float time;
        cudaEventElapsedTime(&time, start_, stop_);
        return time * 1e-3;
    }
    private:
    cudaEvent_t start_, stop_;
};


void checkGpuMem(const char* str="  ")

{
    float free_m,total_m,used_m;
    size_t free_t,total_t;
    cudaMemGetInfo(&free_t,&total_t);
    
    free_m = (uint64_t)free_t/1048576.0 ;
    total_m = (unsigned long long)total_t/1048576.0;
    used_m = total_m-free_m;
    
    printf("  (%s)mem free %f MB mem total %f MB mem used %f MB\n", str, free_m, total_m, used_m);
    fflush(stdout);
}

__global__ void init_data(cuFloatComplex* arr, int n) {
    uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        arr[i].x = (float)i;
        arr[i].y = (float)0;
    }
}

__global__ void check_out(cuFloatComplex* C_gpu, cuFloatComplex* C_d, int n) {
    uint64_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if ( (abs(float(C_gpu[i].x-C_d[i].x))>=1e-5 || abs(float(C_gpu[i].y-C_d[i].y))>=1e-5) ) {
        printf("** index: %llu ** Goes Wrong (%f, %f)\n", i, float(C_gpu[i].x-C_d[i].x), float(C_gpu[i].y-C_d[i].y));
    }
}

struct Contraction {
    Contraction(void *A_d, void *B_d, void *C_d,
                floatTypeCompute alpha, floatTypeCompute beta) :
                A_d_(A_d), B_d_(B_d), C_d_(C_d), alpha_(alpha), beta_(beta) {}
    
    bool init(cudaDataType_t typeA, cudaDataType_t typeB, cudaDataType_t typeC, 
                int nmodeA, int nmodeB, int nmodeC, 
                std::vector<int> modeA, std::vector<int> modeB, std::vector<int> modeC,
                std::vector<int64_t> extentA, std::vector<int64_t> extentB, std::vector<int64_t> extentC
                ) {
        
        /*************************
         * cuTENSOR
         *************************/

        HANDLE_ERROR(cutensorInit(&handle_));
        
        /**********************
         * Create Tensor Descriptors
         **********************/

        HANDLE_ERROR(cutensorInitTensorDescriptor(&handle_,
                     &descA_,
                     nmodeA,
                     extentA.data(),
                     NULL,/*stride*/
                     typeA, CUTENSOR_OP_IDENTITY));

        HANDLE_ERROR(cutensorInitTensorDescriptor(&handle_,
                     &descB_,
                     nmodeB,
                     extentB.data(),
                     NULL,/*stride*/
                     typeB, CUTENSOR_OP_IDENTITY));

        HANDLE_ERROR(cutensorInitTensorDescriptor( &handle_,
                     &descC_,
                     nmodeC,
                     extentC.data(),
                     NULL,/*stride*/
                     typeC, CUTENSOR_OP_IDENTITY));

        /**********************************************
         * Retrieve the memory alignment for each tensor
         **********************************************/ 

         HANDLE_ERROR(cutensorGetAlignmentRequirement(&handle_,
                      A_d_,
                      &descA_,
                      &alignmentRequirementA_));

         HANDLE_ERROR(cutensorGetAlignmentRequirement(&handle_,
                      B_d_,
                      &descB_,
                      &alignmentRequirementB_));

         HANDLE_ERROR(cutensorGetAlignmentRequirement(&handle_,
                      C_d_,
                      &descC_, 
                      &alignmentRequirementC_));

        /*******************************
         * Create Contraction Descriptor
         *******************************/

        HANDLE_ERROR(cutensorInitContractionDescriptor(&handle_, 
                     &desc_,
                     &descA_, modeA.data(), alignmentRequirementA_,
                     &descB_, modeB.data(), alignmentRequirementB_,
                     &descC_, modeC.data(), alignmentRequirementC_,
                     &descC_, modeC.data(), alignmentRequirementC_,
                     typeCompute));

        /**************************
        * Set the algorithm to use
        ***************************/

        HANDLE_ERROR(cutensorInitContractionFind( 
                     &handle_, &find_, 
                     CUTENSOR_ALGO_DEFAULT));

        /**********************
         * Query workspace
         **********************/

        HANDLE_ERROR(cutensorContractionGetWorkspaceSize(&handle_,
                     &desc_,
                     &find_,
                     CUTENSOR_WORKSPACE_RECOMMENDED, &worksize_));

        if (worksize_ > 0)
        {
            if (cudaSuccess != cudaMalloc(&work_, worksize_))
            {
                work_ = nullptr;
                worksize_ = 0;
            }
        } 

        /**************************
         * Create Contraction Plan
         **************************/

        HANDLE_ERROR(cutensorInitContractionPlan(&handle_,
                     &plan_,
                     &desc_,
                     &find_,
                     worksize_));
                     
        return true;
    }
    
    bool execute() {
        cutensorStatus_t err = cutensorContraction(&handle_,
                                  &plan_,
                                  (void*) &alpha_, A_d_, B_d_,
                                  (void*) &beta_,  C_d_, C_d_, 
                                  work_, worksize_, 0 /* stream */);
        if (err != CUTENSOR_STATUS_SUCCESS)
        {
            printf("ERROR: %s in line %d\n", cutensorGetErrorString(err), __LINE__);
        }
        
        return true;
    }
    
    
private:

    void *A_d_;
    void *B_d_;
    void *C_d_;
    floatTypeCompute alpha_;
    floatTypeCompute beta_;
    
    cutensorHandle_t handle_;
    cutensorTensorDescriptor_t descA_;
    cutensorTensorDescriptor_t descB_;
    cutensorTensorDescriptor_t descC_;
    uint32_t alignmentRequirementA_;
    uint32_t alignmentRequirementB_;
    uint32_t alignmentRequirementC_;
    cutensorContractionDescriptor_t desc_;
    cutensorContractionFind_t find_;
    uint64_t worksize_ = 0;
    void *work_ = nullptr;
    cutensorContractionPlan_t plan_;
};

int main()
{
    floatTypeCompute alpha = make_cuFloatComplex(1.0f, 0.f) ;
    floatTypeCompute beta  = make_cuFloatComplex(0.f, 0.f);

    /**********************
     * Computing: C_{m,u,n,v} = alpha * A_{m,h,k,n} B_{u,k,v,h} + beta * C_{m,u,n,v}
     **********************/

    std::vector<int> modeA{'u', 'm'};
    std::vector<int> modeB{'m', 'h'};
    std::vector<int> modeC{'u', 'h'};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;
    extent['m'] = 1 << 11;
    extent['n'] = 2;
    extent['u'] = 1 << 30;
    extent['v'] = 2;
    extent['h'] = 2;
    extent['k'] = 2;

    double gflops = (2.0 * extent['m'] * extent['n'] * extent['u'] * extent['v'] * extent['k'] * extent['h']) /1e9;

    std::vector<int64_t> extentC;
    for (auto mode : modeC)
        extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA;
    for (auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentB;
    for (auto mode : modeB)
        extentB.push_back(extent[mode]);

    /**********************
     * Allocating data
     **********************/

    size_t elementsA = 1;
    for (auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsB = 1;
    for (auto mode : modeB)
        elementsB *= extent[mode];
    size_t elementsC = 1;
    for (auto mode : modeC)
        elementsC *= extent[mode];

    uint64_t sizeA = sizeof(floatTypeA) * elementsA;
    uint64_t sizeB = sizeof(floatTypeB) * elementsB;
    uint64_t sizeC = sizeof(floatTypeC) * elementsC;
    printf("Total memory: %.2f GiB\n", (sizeA + sizeB + sizeC)/1024./1024./1024);
    fflush(stdout);

    checkGpuMem("before cudaMalloc");
    
    // unified memory
    void *A_d, *B_d, *C_d;
    HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &A_d, sizeA));
    HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &B_d, sizeB));
    HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &C_d, sizeC));
    
    checkGpuMem("after cudaMalloc");

    /*******************
     * Initialize data
     *******************/
    
    init_data<<<(elementsA+255)/256, 256>>>((floatTypeA*)A_d, elementsA);
    init_data<<<(elementsB+255)/256, 256>>>((floatTypeB*)B_d, elementsB);
    init_data<<<(elementsC+255)/256, 256>>>((floatTypeC*)C_d, elementsC);
    cudaDeviceSynchronize();
    printf("***Finish Initializing***\n");
    
    /*************************
     * cuTENSOR
     *************************/ 

    Contraction contraction = Contraction(A_d, B_d, C_d, alpha, beta);
    contraction.init(typeA, typeB, typeC, 
                    nmodeA, nmodeB, nmodeC, 
                    modeA, modeB, modeC,
                    extentA, extentB, extentC);
    
    /**********************
     * Run
     **********************/

    checkGpuMem("before contraction");

    double minTimeCUTENSOR = 1e100;
    for (int i=0; i < 3; ++i)
    {
        // Set up timing
        GPUTimer timer;
        timer.start();

        contraction.execute();

        // Synchronize and measure timing
        auto time = timer.seconds();
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
    }
    
    checkGpuMem("after contraction");
    
    /**********************
     * Check The Ein
     **********************/
    printf("nmodeA: %d\n", nmodeA);
    
    printf("extentA:");
    for (int64_t i: extentA) {
        printf("%lld\t", i);
    }
    printf("\n");
    
    printf("modeA:");
    for (int i: modeA) {
        printf("%c\t", char(i));
    }
    printf("\n");
    
    
    printf("nmodeB: %d\n", nmodeB);
    
    printf("extentB:");
    for (int64_t i: extentB) {
        printf("%lld\t", i);
    }
    printf("\n");
    
    printf("modeB:");
    for (int i: modeB) {
        printf("%c\t", char(i));
    }
    printf("\n");
    
    
    printf("nmodeC: %d\n", nmodeC);
    
    printf("extentC:");
    for (int64_t i: extentC) {
        printf("%lld\t", i);
    }
    printf("\n");
    
    printf("modeC:");
    for (int i: modeC) {
        printf("%c\t", char(i));
    }
    printf("\n");

    /*************************/

    double transferedBytes = sizeC + sizeA + sizeB;
    transferedBytes += ((float) beta.x != 0.f) ? sizeC : 0;
    transferedBytes /= 1e9;
    printf("cuTensor: %.2f GFLOPs/s %.2f GB/s\n", gflops / minTimeCUTENSOR, transferedBytes/ minTimeCUTENSOR);
    
    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);

    return 0;
}
