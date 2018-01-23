#include <iostream>
#include <chrono>
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

