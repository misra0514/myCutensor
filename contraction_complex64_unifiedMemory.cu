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

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>
#include <cuComplex.h>
#include <curand_kernel.h>

#include "Val.h"


typedef cuFloatComplex floatTypeA;
typedef cuFloatComplex floatTypeB;
typedef cuFloatComplex floatTypeC;
typedef cuFloatComplex floatTypeCompute;

cudaDataType_t typeA = CUDA_C_32F;
cudaDataType_t typeB = CUDA_C_32F;
cudaDataType_t typeC = CUDA_C_32F;
cutensorComputeType_t typeCompute = CUTENSOR_COMPUTE_TF32;

uint64_t BATCH = 2;
uint64_t SECOND_EXTENT = 2;


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


struct EinInfo {
    EinInfo(std::vector<int> _modeA, std::vector<int> _modeB, std::vector<int> _modeC, char batch_mode='p') {
        
        modeA = _modeA;
        modeB = _modeB;
        modeC = _modeC;
        std::unordered_map<int, uint64_t> extent;
        for (auto it = modeA.begin(); it != modeA.end(); it++) {
            if (*it == 'p') {
                extent[*it] = BATCH;
            }
            else if (*it == 'm') {
                extent[*it] = SECOND_EXTENT;
            }
            else extent[*it] = 2;
        }
        for (auto it = modeB.begin(); it != modeB.end(); it++) {
            if (*it == 'p') {
                extent[*it] = BATCH;
            }
            else if (*it == 'm') {
                extent[*it] = SECOND_EXTENT;
            }
            else extent[*it] = 2;
        }
        for (auto it = modeC.begin(); it != modeC.end(); it++) {
            if (*it == 'p') {
                extent[*it] = BATCH;
            }
            else if (*it == 'm') {
                extent[*it] = SECOND_EXTENT;
            }
            else extent[*it] = 2;
        }
        
        /* get modeA, modeB, modeC */
        std::reverse(modeA.begin(), modeA.end());
        std::reverse(modeB.begin(), modeB.end());
        std::reverse(modeC.begin(), modeC.end());
        
        /* get nmodeA, nmodeB, nmodeC */
        nmodeA = modeA.size();
        nmodeB = modeB.size();
        nmodeC = modeC.size();
        
        /* get extentA, extentB, extentC */
        for (auto mode : modeA)
            extentA.push_back(extent[mode]);
        for (auto mode : modeB)
            extentB.push_back(extent[mode]);
        for (auto mode : modeC)
            extentC.push_back(extent[mode]);
            
        /* get elementA, elementB, elementC */
        for (auto mode : modeA)
            elementsA *= extent[mode];
        for (auto mode : modeB)
            elementsB *= extent[mode];
        for (auto mode : modeC)
            elementsC *= extent[mode];
            
        /* get sizeA, sizeB, sizeC */
        sizeA = sizeof(cuFloatComplex) * elementsA;
        sizeB = sizeof(cuFloatComplex) * elementsB;
        sizeC = sizeof(cuFloatComplex) * elementsC;
    }
    
public:
    std::vector<int> modeA;
    std::vector<int> modeB;
    std::vector<int> modeC;
    
    int nmodeA;
    int nmodeB;
    int nmodeC;
    
    std::vector<int64_t> extentA;
    std::vector<int64_t> extentB;
    std::vector<int64_t> extentC;
    
    uint64_t elementsA = 1;
    uint64_t elementsB = 1;
    uint64_t elementsC = 1;
    
    uint64_t sizeA;
    uint64_t sizeB;
    uint64_t sizeC;
};


int main()
{

    /**********************
     * Get Info Of The Ein: pmklwjaBYITCLWOUAJDHPNMEVKGXSRFd,oigysvQKGXSRFd->pmklwjoigysvaBYITCLWOUAJDHPNMEVQ
     **********************/

    std::vector<int> modeA{'p','m','k','l','w','j','a','B','Y','I','T','C','L','W','O','U','A','J','D','H','P','N','M','E','V','K','G','X','S','R','F','d'};
    std::vector<int> modeB{'o','i','g','y','s','v','Q','K','G','X','S','R','F','d'};
    std::vector<int> modeC{'p','m','k','l','w','j','o','i','g','y','s','v','a','B','Y','I','T','C','L','W','O','U','A','J','D','H','P','N','M','E','V','Q'};
    EinInfo info_unified(modeA, modeB, modeC); 
    
    std::vector<int> modeA_gpu{'m','k','l','w','j','a','B','Y','I','T','C','L','W','O','U','A','J','D','H','P','N','M','E','V','K','G','X','S','R','F','d'};
    std::vector<int> modeB_gpu{'o','i','g','y','s','v','Q','K','G','X','S','R','F','d'};
    std::vector<int> modeC_gpu{'m','k','l','w','j','o','i','g','y','s','v','a','B','Y','I','T','C','L','W','O','U','A','J','D','H','P','N','M','E','V','Q'};
    EinInfo info_gpu(modeA_gpu, modeB_gpu, modeC_gpu);
    
    /*************************
     * cuTENSOR
     *************************/  

    Val val = Val();
                    
    // Global Memory
    val.cudaMalloc_init(info_gpu.sizeA, info_gpu.sizeB, info_gpu.sizeC, 
                    info_gpu.elementsA, info_gpu.elementsB, info_gpu.elementsC);
    checkGpuMem("after cudaMalloc_init");
    val.cal_gpu(typeA, typeB, typeC, typeCompute,
                    info_gpu.nmodeA, info_gpu.nmodeB, info_gpu.nmodeC,
                    info_gpu.modeA, info_gpu.modeB, info_gpu.modeC,
                    info_gpu.extentA, info_gpu.extentB, info_gpu.extentC);
    
    // Unified Memory
    val.cudaMallocManaged_init(info_unified.sizeA, info_unified.sizeB, info_unified.sizeC, 
                                info_unified.elementsA, info_unified.elementsB, info_unified.elementsC);
    checkGpuMem("after cudaMallocManaged_init");
    printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    for (int algo = -6; algo < 109; algo++) {
        printf("    >>>>>>>>>>>>> algo: %d >>>>>>>>>>>\n", algo);
        val.cal_unified(typeA, typeB, typeC, typeCompute,
                    info_unified.nmodeA, info_unified.nmodeB, info_unified.nmodeC,
                    info_unified.modeA, info_unified.modeB, info_unified.modeC,
                    info_unified.extentA, info_unified.extentB, info_unified.extentC, 
                    algo);
        
        val.check_out(info_unified.elementsC, info_gpu.elementsC);
        printf("\n");
    }
    printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");

       
    /**********************
     * Check The Ein
     **********************/
    printf("nmodeA: %d\n", info_unified.nmodeA);
    
    printf("extentA:");
    for (int64_t i: info_unified.extentA) {
        printf("%lld ", i);
    }
    printf("\n");
    
    printf("modeA:");
    for (int i: info_unified.modeA) {
        printf("%c ", char(i));
    }
    printf("\n");
    
    
    printf("nmodeB: %d\n", info_unified.nmodeB);
    
    printf("extentB:");
    for (int64_t i: info_unified.extentB) {
        printf("%lld ", i);
    }
    printf("\n");
    
    printf("modeB:");
    for (int i: info_unified.modeB) {
        printf("%c ", char(i));
    }
    printf("\n");
    
    
    printf("nmodeC: %d\n", info_unified.nmodeC);
    
    printf("extentC:");
    for (int64_t i: info_unified.extentC) {
        printf("%lld ", i);
    }
    printf("\n");
    
    printf("modeC:");
    for (int i: info_unified.modeC) {
        printf("%c ", char(i));
    }
    printf("\n");

    return 0;
}
