#include "nestedReduce.h"

__global__ void neighboredReduce(int* g_idata,int* g_odata,int size) 
{
    auto tid  = threadIdx.x;
    auto g_idx = threadIdx.x + blockIdx.x*blockDim.x;

    for(unsigned int stride = 1; stride < blockDim.x;stride *= 2)
    {
        if(tid % (2*stride) == 0)
        {
            g_idata[g_idx] += g_idata[g_idx + stride];
        }
        __syncthreads();
    }
    if(tid == 0) g_odata[blockIdx.x] = g_idata[g_idx];
}

void call_neighbored_reduce( int* g_idata,  int* g_odata,dim3 block,dim3 grid)
{
    neighboredReduce<<<grid,block>>>(g_idata,g_odata,block.x);
}

__global__ void gpuRecursiveReduce(int* g_idata, int* g_odata,int size) 
{
    auto tid = threadIdx.x;
    auto g_idx = threadIdx.x + blockDim.x*blockIdx.x;

    if(size == 2 && tid == 0)
    {
        g_odata[tid] = g_idata[0] + g_idata[1]; 
        return ;
    }
    int stride = size / 2;
    if(stride > 1 && tid < stride)
    {
        g_idata[tid] += g_idata[tid + stride];
    }
    __syncthreads();
    if(tid == 0)
    {
        gpuRecursiveReduce<<<1,stride>>>(g_idata + g_idx,g_odata + blockIdx.x,stride);
        cudaDeviceSynchronize();
    }
    __syncthreads();
}

void call_recursive_reduce(int* g_idata, int* g_odata,int block,int grid)
{
    gpuRecursiveReduce<<<grid,block>>>(g_idata,g_odata,block.x);
}

