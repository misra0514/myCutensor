# cuTENSOR - Samples#

* [Documentation](https://docs.nvidia.com/cuda/cutensor/index.html)

# Install

## Linux 

With cmake

```
mkdir build && cd build
cmake .. -DCUTENSOR_ROOT=<path_to_cutensor_root>
make -j8

srun -p 1005test --quotatype=spot --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 ./contraction_complex64_unifiedMemory    
```
