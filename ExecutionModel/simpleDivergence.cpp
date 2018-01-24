#include <chrono>
#include <iostream>
#include <stdlib.h>
#include "simple.h"
#include "simpleDivergence.h"
#include "cuda_runtime.h"
#include "../Commen/commen.h"


void simpleDivergence(int argc,char** argv)
{
	int block_size = 64;
	int total_size = 64;
	if (argc > 1) {
		total_size = atoi(argv[1]);
	}
	if (argc > 2) {
		block_size = atoi(argv[2]);
	}
	float* c = nullptr;
	auto release = [&] (){
		cudaFree(c);
	};
	
	CHECK_WITH_RELEASED(cudaMalloc(&c, total_size),release);
	auto start = CLK;
	call_warmingup(c, total_size, block_size);
	cudaDeviceSynchronize();
	std::cout << "warmingup elapsed:" << MS(CLK,start)<<" ms" << std::endl;

	start = CLK;
	call_mat_kernel1(c, total_size, block_size);
	cudaDeviceSynchronize();
	std::cout << "mat_kernel1 elapsed:" << MS(CLK,start)<<" ms" << std::endl;

	start = CLK;
	call_mat_kernel2(c, total_size, block_size);
	cudaDeviceSynchronize();
	std::cout << "mat_kernel2 elapsed:" << MS(CLK,start)<<" ms"  << std::endl;

	start = CLK;
	call_mat_kernel3(c, total_size, block_size);
	cudaDeviceSynchronize();
	std::cout << "mat_kernel3 elapsed:" << MS(CLK,start)<<" ms"  << std::endl;

	start = CLK;
	call_mat_kernel4(c, total_size, block_size);
	cudaDeviceSynchronize();
	std::cout << "mat_kernel4 elapsed:" << MS(CLK,start)<<" ms"  << std::endl;
}