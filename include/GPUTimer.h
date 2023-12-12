#pragma once


struct GPUTimer
{
    GPUTimer();

    ~GPUTimer();

    void start();

    float seconds();
        
private:
    cudaEvent_t start_, stop_;
};