#pragma once


namespace multiBatch {

bool cudaMallocManaged_init(void** A_d_t, void** B_d_t, void** C_d_t, 
                            uint64_t sizeA, uint64_t sizeB, uint64_t sizeC, 
                            uint64_t elementsA, uint64_t elementsB, uint64_t elementsC);
    
bool cudaMallocManaged_ABC(void** A_d_t, void** B_d_t, void** C_d_t, 
                            uint64_t sizeA, uint64_t sizeB, uint64_t sizeC);
   
void init_unified_ABC(void* A_d, void* B_d, void* C_d, 
                      uint64_t elementsA, uint64_t elementsB, uint64_t elementsC);
    
bool cudaMalloc_init(void** A_gpu_t, void** B_gpu_t, void** C_gpu_t, 
                    uint64_t sizeA, uint64_t sizeB, uint64_t sizeC, 
                    uint64_t elementsA, uint64_t elementsB, uint64_t elementsC);
   
void cal_unified(void *A_d, void *B_d, void *C_d, 
                    cudaDataType_t typeA, cudaDataType_t typeB, cudaDataType_t typeC, cutensorComputeType_t typeCompute,
                    int nmodeA, int nmodeB, int nmodeC,
                    std::vector<int> modeA, std::vector<int> modeB, std::vector<int> modeC,
                    std::vector<int64_t> extentA, std::vector<int64_t> extentB, std::vector<int64_t> extentC,
                    int algo);
    
void cal_gpu(void *A_gpu, void *B_gpu, void *C_gpu, 
                    cudaDataType_t typeA, cudaDataType_t typeB, cudaDataType_t typeC, cutensorComputeType_t typeCompute,
                    int nmodeA, int nmodeB, int nmodeC,
                    std::vector<int> modeA, std::vector<int> modeB, std::vector<int> modeC,
                    std::vector<int64_t> extentA, std::vector<int64_t> extentB, std::vector<int64_t> extentC,
                    int algo);
    
void check_out(void *C_gpu, void *C_d, uint64_t info_unified_elementsC, uint64_t info_gpu_elementsC);
} // namespace multiBatch
    
struct Val {
    Val();
        
    bool cudaMallocManaged_init(uint64_t sizeA, uint64_t sizeB, uint64_t sizeC, 
                                uint64_t elementsA, uint64_t elementsB, uint64_t elementsC);
    
    bool cudaMalloc_init(uint64_t sizeA, uint64_t sizeB, uint64_t sizeC, 
                        uint64_t elementsA, uint64_t elementsB, uint64_t elementsC);
    
    //void init_unified(uint64_t elementsA, uint64_t elementsB, uint64_t elementsC);
    
    //void init_gpu(uint64_t elementsA, uint64_t elementsB, uint64_t elementsC);
        
    void cal_unified(cudaDataType_t typeA, cudaDataType_t typeB, cudaDataType_t typeC, cutensorComputeType_t typeCompute,
                     int nmodeA, int nmodeB, int nmodeC,
                     std::vector<int> modeA, std::vector<int> modeB, std::vector<int> modeC,
                     std::vector<int64_t> extentA, std::vector<int64_t> extentB, std::vector<int64_t> extentC,
                     int algo=-1);
    
    void cal_gpu(cudaDataType_t typeA, cudaDataType_t typeB, cudaDataType_t typeC, cutensorComputeType_t typeCompute,
                 int nmodeA, int nmodeB, int nmodeC,
                 std::vector<int> modeA, std::vector<int> modeB, std::vector<int> modeC,
                 std::vector<int64_t> extentA, std::vector<int64_t> extentB, std::vector<int64_t> extentC,
                 int algo=-1);
        
    void check_out(uint64_t info_unified_elementsC, uint64_t info_gpu_elementsC);
    
    ~Val();
        
private:
    
    void *A_d_;
    void *B_d_;
    void *C_d_;
    
    void *A_gpu_;
    void *B_gpu_;
    void *C_gpu_;
};