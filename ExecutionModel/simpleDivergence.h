/**************************************************************************
Author		: xuwj
Date                          : 2018/01/23 23:02
Description	: simpleDivergence头文件
**************************************************************************/
#ifndef SIMPLEDIVERGENCE_H_
#define SIMPLEDIVERGENCE_H_
/*
线程束分化测试，在gtx 750和gtx 940m上测试这几种方法的分支效率都是100%，没有发生线程束分化
*/
void call_mat_kernel1(float*c, int size, int block_size);
void call_mat_kernel2(float*c, int size, int block_size);
void call_mat_kernel3(float*c, int size, int block_size);
void call_mat_kernel4(float*c, int size, int block_size);
void call_warmingup(float* c, int size, int block_size);
#endif //SIMPLEDIVERGENCE_H_