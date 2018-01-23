#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void addWithCuda(float* mat_a,float* mat_b,float* mat_c,int nx,int ny);
