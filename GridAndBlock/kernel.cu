#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void addKernel(float *c, const float *a, const float *b,int nx,int ny)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
    int i = y*nx + x;
	if (y < ny && x < nx)
	{
		c[i] = a[i] + b[i];
	}
}

// Helper function for using CUDA to add vectors in parallel.
void addWithCuda(float* mat_a,float* mat_b,float* mat_c,int nx,int ny)
{
    dim3 block(32, 32);	
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    addKernel<<<grid, block>>>(mat_c,mat_a,mat_b,nx,ny);
}
