#include <stdlib.h>
#include <stdio.h>

#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>
#include <cuComplex.h>
#include <curand_kernel.h>

#include "Contraction.h"

#define HANDLE_ERROR(x)                                               \
{ const auto err = x;                                                 \
  if( err != CUTENSOR_STATUS_SUCCESS )                                \
  { printf("Error: %s\n", cutensorGetErrorString(err)); return err; } \
};

Contraction::Contraction(void *A_d, void *B_d, void *C_d,
                         cuFloatComplex alpha, cuFloatComplex beta) :
                        A_d_(A_d), B_d_(B_d), C_d_(C_d), alpha_(alpha), beta_(beta) {}
    
bool Contraction::init(cudaDataType_t typeA, cudaDataType_t typeB, cudaDataType_t typeC, cutensorComputeType_t typeCompute, 
                int nmodeA, int nmodeB, int nmodeC, 
                std::vector<int> modeA, std::vector<int> modeB, std::vector<int> modeC,
                std::vector<int64_t> extentA, std::vector<int64_t> extentB, std::vector<int64_t> extentC, 
                int algo) {
        
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
    
    //printf("CUTENSOR_ALGO_DEFAULT_PATIENT: %d\n", (int) CUTENSOR_ALGO_DEFAULT_PATIENT);
    //printf("CUTENSOR_ALGO_GETT: %d\n", (int) CUTENSOR_ALGO_GETT);
    //printf("CUTENSOR_ALGO_TGETT: %d\n", (int) CUTENSOR_ALGO_TGETT);
    //printf("CUTENSOR_ALGO_TTGT: %d\n", (int) CUTENSOR_ALGO_TTGT);
    //printf("CUTENSOR_ALGO_DEFAULT: %d\n", (int) CUTENSOR_ALGO_DEFAULT);
    
    //int32_t maxAlgosTC = 0;
    //cutensorContractionMaxAlgos(&maxAlgosTC);
    //printf("maxAlgosTC: %d\n", (int)maxAlgosTC);
    //printf(" using the ALGO: %d\n", algo);
    
    
    /**************************
     * Set the algorithm to use
     **************************/

    cutensorContractionFind_t find;
    HANDLE_ERROR(cutensorInitContractionFind( 
                 &handle_, &find, 
                 (cutensorAlgo_t)algo));


    /**************************
     * Create Contraction Plan
     **************************/

    HANDLE_ERROR(cutensorInitContractionPlan(&handle_,
                 &plan_,
                 &desc_,
                 &find,
                 worksize_));

    return true;
}
    
bool Contraction::execute() {
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
