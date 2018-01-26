#include <chrono>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include "simple.h"
#include "simpleDivergence.h"
#include "cuda_runtime.h"
#include "../Commen/commen.h"


void simpleDivergence(int argc,char** argv)
{
	int block_size = 64;
	int grid_size = 64;
	if (argc > 1) {
		grid_size = atoi(argv[1]);
	}
	if (argc > 2) {
		block_size = atoi(argv[2]);
	}
	float* c = nullptr;
	auto release = [&] (){
		cudaFree(c);
	};
	std::vector<float> res1(grid_size*block_size);
	std::vector<float> res2(grid_size*block_size);
	std::vector<float> res3(grid_size*block_size);
	std::vector<float> res4(grid_size*block_size);
	
	
	CHECK_WITH_RELEASED(cudaMalloc(&c, grid_size*block_size),release);
	auto start = CLK;
	call_warmingup(c, grid_size, block_size);
	cudaDeviceSynchronize();
	std::cout << "warmingup elapsed:" << MS(CLK,start)<<" ms" << std::endl;

	start = CLK;
	call_mat_kernel1(c, grid_size, block_size);
	cudaDeviceSynchronize();
	std::cout << "mat_kernel1 elapsed:" << MS(CLK,start)<<" ms" << std::endl;
	cudaMemcpy(&res1[0], c, res1.size(),cudaMemcpyDeviceToHost);

	start = CLK;
	call_mat_kernel2(c, grid_size, block_size);
	cudaDeviceSynchronize();
	std::cout << "mat_kernel2 elapsed:" << MS(CLK,start)<<" ms"  << std::endl;
	cudaMemcpy(&res2[0], c, res1.size(),cudaMemcpyDeviceToHost);

	start = CLK;
	call_mat_kernel3(c, grid_size, block_size);
	cudaDeviceSynchronize();
	std::cout << "mat_kernel3 elapsed:" << MS(CLK,start)<<" ms"  << std::endl;
	cudaMemcpy(&res3[0], c, res1.size(),cudaMemcpyDeviceToHost);

	start = CLK;
	call_mat_kernel4(c, grid_size, block_size);
	cudaDeviceSynchronize();
	std::cout << "mat_kernel4 elapsed:" << MS(CLK,start)<<" ms"  << std::endl;
	cudaMemcpy(&res4[0], c, res1.size(),cudaMemcpyDeviceToHost);
	return;
}