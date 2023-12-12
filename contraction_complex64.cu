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

int main()
{
    typedef cuFloatComplex floatTypeA;
    typedef cuFloatComplex floatTypeB;
    typedef cuFloatComplex floatTypeC;
    typedef cuFloatComplex floatTypeCompute;

    cudaDataType_t typeA = CUDA_C_32F;
    cudaDataType_t typeB = CUDA_C_32F;
    cudaDataType_t typeC = CUDA_C_32F;
    cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_32F;

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
    extent['m'] = 2;
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

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeB = sizeof(floatTypeB) * elementsB;
    size_t sizeC = sizeof(floatTypeC) * elementsC;
    printf("Total memory: %.2f GiB\n", (sizeA + sizeB + sizeC)/1024./1024./1024);

    void *A_d, *B_d, *C_d;
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &A_d, sizeA));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &B_d, sizeB));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &C_d, sizeC));

    floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
    floatTypeB *B = (floatTypeB*) malloc(sizeof(floatTypeB) * elementsB);
    floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);

    if (A == NULL || B == NULL || C == NULL)
    {
        printf("Error: Host allocation of A or C.\n");
        return -1;
    }

    /*******************
     * Initialize data
     *******************/

    for (int64_t i = 0; i < elementsA; i++){
        // A[i].x = (((float) rand())/RAND_MAX - 0.5)*100;
        // A[i].y = (((float) rand())/RAND_MAX - 0.5)*100;
        A[i].x = (float)i;
        A[i].y = (float)0;
    }
        
    for (int64_t i = 0; i < elementsB; i++){
        // B[i].x = (((float) rand())/RAND_MAX - 0.5)*100;
        // B[i].y = (((float) rand())/RAND_MAX - 0.5)*100;
        B[i].x = (float)i;
        B[i].y = (float)0;
    }
        
    for (int64_t i = 0; i < elementsC; i++){
        // C[i].x = (((float) rand())/RAND_MAX - 0.5)*100;
        // C[i].y = (((float) rand())/RAND_MAX - 0.5)*100;
        C[i].x = (float)0;
        C[i].y = (float)0;
    }
        

    HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));

    /*************************
     * cuTENSOR
     *************************/ 

    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorInit(&handle));

    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
                 &descA,
                 nmodeA,
                 extentA.data(),
                 NULL,/*stride*/
                 typeA, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descB;
    HANDLE_ERROR(cutensorInitTensorDescriptor(&handle,
                 &descB,
                 nmodeB,
                 extentB.data(),
                 NULL,/*stride*/
                 typeB, CUTENSOR_OP_IDENTITY));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorInitTensorDescriptor( &handle,
                 &descC,
                 nmodeC,
                 extentC.data(),
                 NULL,/*stride*/
                 typeC, CUTENSOR_OP_IDENTITY));

    /**********************************************
     * Retrieve the memory alignment for each tensor
     **********************************************/ 

     uint32_t alignmentRequirementA;
     HANDLE_ERROR(cutensorGetAlignmentRequirement(&handle,
                  A_d,
                  &descA,
                  &alignmentRequirementA));

     uint32_t alignmentRequirementB;
     HANDLE_ERROR(cutensorGetAlignmentRequirement(&handle,
                  B_d,
                  &descB,
                  &alignmentRequirementB));

     uint32_t alignmentRequirementC;
     HANDLE_ERROR(cutensorGetAlignmentRequirement(&handle,
                  C_d,
                  &descC, 
                  &alignmentRequirementC));

    /*******************************
     * Create Contraction Descriptor
     *******************************/

    cutensorContractionDescriptor_t desc;
    HANDLE_ERROR(cutensorInitContractionDescriptor(&handle, 
                 &desc,
                 &descA, modeA.data(), alignmentRequirementA,
                 &descB, modeB.data(), alignmentRequirementB,
                 &descC, modeC.data(), alignmentRequirementC,
                 &descC, modeC.data(), alignmentRequirementC,
                 typeCompute));

    /**************************
    * Set the algorithm to use
    ***************************/

    cutensorContractionFind_t find;
    HANDLE_ERROR(cutensorInitContractionFind( 
                 &handle, &find, 
                 CUTENSOR_ALGO_DEFAULT));

    /**********************
     * Query workspace
     **********************/

    uint64_t worksize = 0;
    HANDLE_ERROR(cutensorContractionGetWorkspaceSize(&handle,
                 &desc,
                 &find,
                 CUTENSOR_WORKSPACE_RECOMMENDED, &worksize));

    void *work = nullptr;
    if (worksize > 0)
    {
        if (cudaSuccess != cudaMalloc(&work, worksize))
        {
            work = nullptr;
            worksize = 0;
        }
    } 

    /**************************
     * Create Contraction Plan
     **************************/

    cutensorContractionPlan_t plan;
    HANDLE_ERROR(cutensorInitContractionPlan(&handle,
                 &plan,
                 &desc,
                 &find,
                 worksize));

    /**********************
     * Run
     **********************/

    double minTimeCUTENSOR = 1e100;
    cutensorStatus_t err;
    for (int i=0; i < 3; ++i)
    {
        cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        // Set up timing
        GPUTimer timer;
        timer.start();

        err = cutensorContraction(&handle,
                                  &plan,
                                  (void*) &alpha, A_d, B_d,
                                  (void*) &beta,  C_d, C_d, 
                                  work, worksize, 0 /* stream */);

        // Synchronize and measure timing
        auto time = timer.seconds();

        if (err != CUTENSOR_STATUS_SUCCESS)
        {
            printf("ERROR: %s in line %d\n", cutensorGetErrorString(err), __LINE__);
        }
        minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
    }
    
    /**********************
     * Check The Output
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
    
    
        /***************************/
    
    
    printf("A:");
    for (int64_t i = 0; i < elementsA; i++) {
         printf("%f\t", A[i].x);
         printf("%f\t\t", A[i].y);
    }
    printf("\n");

    printf("B:");
    for (int64_t i = 0; i < elementsB; i++) {
         printf("%f\t", B[i].x);
         printf("%f\t\t", B[i].y);
    }
    printf("\n");
    
    HANDLE_CUDA_ERROR(cudaMemcpy(C, C_d, sizeC, cudaMemcpyDeviceToHost));
    printf("C:");
    for (int64_t i = 0; i < elementsC; i++) {
         printf("%f\t", C[i].x);
         printf("%f\t\t", C[i].y);
    }
    printf("\n");
    /*************************/

    double transferedBytes = sizeC + sizeA + sizeB;
    transferedBytes += ((float) beta.x != 0.f) ? sizeC : 0;
    transferedBytes /= 1e9;
    printf("cuTensor: %.2f GFLOPs/s %.2f GB/s\n", gflops / minTimeCUTENSOR, transferedBytes/ minTimeCUTENSOR);

    if (A) free(A);
    if (B) free(B);
    if (C) free(C);
    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (work) cudaFree(work);

    return 0;
}
