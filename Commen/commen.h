/**************************************************************************
Author		: xuwj
Date                          : 2018/01/23 23:58
Description	: 
**************************************************************************/
#ifndef COMMEN_H_
#define COMMEN_H_
#include <iostream>
#include <chrono>
#include <iostream>
#define CLK  std::chrono::high_resolution_clock()

#define CHECK(call) {\
	const cudaError_t err = call; \
	if (err !=  cudaSuccess) \
	{\
		std::cout<<"error:"<<__FILE__<<","<<__LINE__<<","<<cudaGetErrorString(err)<<std::endl; \
	} \
}

#define CHECK_WITH_RELEASED(call,relesed) {\
	const cudaError_t err = call; \
	if (err != cudaSuccess) \
	{\
	std::cout << "error:" << __FILE__ << "," << __LINE__ << "," << cudaGetErrorString(err) << std::endl; \
	relesed();\
	} \
}


#endif //COMMEN_H_
