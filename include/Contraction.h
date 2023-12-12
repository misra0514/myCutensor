#pragma once

struct Contraction {
    Contraction(void *A_d, void *B_d, void *C_d,
                cuFloatComplex alpha, cuFloatComplex beta);

    bool init(cudaDataType_t typeA, cudaDataType_t typeB, cudaDataType_t typeC, cutensorComputeType_t typeCompute,
                int nmodeA, int nmodeB, int nmodeC,
                std::vector<int> modeA, std::vector<int> modeB, std::vector<int> modeC,
                std::vector<int64_t> extentA, std::vector<int64_t> extentB, std::vector<int64_t> extentC, 
                int algo);

    bool execute();

private:

    void *A_d_;
    void *B_d_;
    void *C_d_;
    cuFloatComplex alpha_;
    cuFloatComplex beta_;

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
