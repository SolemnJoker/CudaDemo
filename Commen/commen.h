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
//Ê±¼ä

#define CLK  std::chrono::high_resolution_clock::now()
#define MS(end,start) ((end - start ).count()/10000000.0)
#define PTIME(start,info) std::cout<<info<<" :"<<MS(CLK,start)<<" ms"<<std::endl;

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

//vector
#define VPTR(v) &v[0]
#define VSIZE(v) v.size()*sizeof(v[0])

#endif //COMMEN_H_
