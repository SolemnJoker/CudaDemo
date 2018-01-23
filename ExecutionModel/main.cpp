#include <stdlib.h>
#include <iostream>
#include <chrono>
#include "cuda_runtime.h"
#include "simpleDivergence.h"
#include "../Commen/commen.h"
int main(int argc,char** argv) {

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
	//std::cout << "warmingup elapsed:" << CLK - start << std::endl;


}