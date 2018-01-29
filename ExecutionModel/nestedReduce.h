/**************************************************************************
Copyright	: Cloudream Inc All Rights Reserved
Author		: xuwj
Date        : 2018/01/29 11:50
Description	: 
**************************************************************************/
#ifndef NESTEDREDUCE_H_
#define NESTEDREDUCE_H_
#include "cuda_runtime.h"
void call_neighbored_reduce( int* g_idata, int* g_odata,dim3 block,dim3 grid);
void call_recursive_reduce( int* g_idata, int* g_odata,dim3 block,dim3 grid);
#endif //NESTEDREDUCE_H_
