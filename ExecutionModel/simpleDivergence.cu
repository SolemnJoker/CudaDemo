#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "simpleDivergence.h"


__global__ void mat_kernel1(float * c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float a = 0;
    float b = 0;
    if(i %2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }
    c[i] = a + b;
}

void call_mat_kernel1(float*c ,int size,int block_size){
    dim3 block(block_size,1);
    dim3 grid(size,1);
    mat_kernel1<<<grid,block>>>(c);
}

__global__ void mat_kernel2(float * c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float a = 0;
    float b = 0;
    if((i/warpSize) %2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }
    c[i] = a + b;
}

void call_mat_kernel2(float*c ,int size,int block_size){
    dim3 block(block_size,1);
    dim3 grid(size,1);
    mat_kernel2<<<grid,block>>>(c);
}


__global__ void warmingup(float* c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float a = 0;
    float b = 0;
    if((i/warpSize) %2 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }
    c[i] = a + b;
}
void call_warmingup(float* c,int size,int block_size){
    dim3 block(block_size,1);
    dim3 grid(size,1);
    warmingup<<<grid,block>>>(c);
}

__global__ void mat_kernel3(float * c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float a = 0;
    float b = 0;
	int ipred = i % 2;
    if(ipred)
    {
        a = 100.0f;
    }
	if(!ipred)
    {
        b = 200.0f;
    }
    c[i] = a + b;
}

void call_mat_kernel3(float*c ,int size,int block_size){
    dim3 block(block_size,1);
    dim3 grid(size,1);
    mat_kernel3<<<grid,block>>>(c);
}


__global__ void mat_kernel4(float * c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float a = 0;
    float b = 0;
	int ipred = i % 2;
    if(ipred)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }
    c[i] = a + b;
}

void call_mat_kernel4(float*c ,int size,int block_size){
    dim3 block(block_size,1);
    dim3 grid(size,1);
    mat_kernel4<<<grid,block>>>(c);
}









