#include "simple.h"
#include "nestedReduce.h"
#include "../Commen/commen.h"

#include <vector>
int cpuRecursiveReduce(int* data, int size)
{
	if (size == 1) return data[0];
	int stride = size / 2;
	for (int i = 0;i < stride;i++)
	{
		data[i] += data[i + stride];
	}
	return cpuRecursiveReduce(data, stride);
}
void initData(std::vector<int>& vec)
{
	for (auto& val:vec)
	{
		val = (int)rand() & 0xFF;
	}
}
void nestedReduce(int argc, char** argv)
{
	int nblock = 2048;
	int nthread = 512;
	int dev = 0;
	cudaDeviceProp prop;
	CHECK(cudaGetDeviceProperties(&prop, dev));
	CHECK(cudaGetDevice(&dev));
	if (argc > 1)
	{
		nblock = atoi(argv[1]);
	}
	if (argc > 2)
	{
		nthread = atoi(argv[2]);
	}
	int nsize = nblock* nthread;
	dim3 block(nblock);
	dim3 grid((nsize + block.x - 1)/block.x,1);
	std::vector<int> h_idata(nsize);
	std::vector<int> h_odata(grid.x);

	int* d_idata = NULL;
	int* d_odata = NULL;
	auto release = [&]() {
		cudaFree(d_idata);
		cudaFree(d_odata);
	};

	CHECK_WITH_RELEASED(cudaMalloc(&d_idata, VSIZE(h_idata)), release);
	CHECK_WITH_RELEASED(cudaMalloc(&d_odata, VSIZE(h_odata)), release);
	CHECK_WITH_RELEASED(cudaMemcpy(d_idata, VPTR(h_idata), VSIZE(h_idata),cudaMemcpyHostToDevice),release);
	CHECK_WITH_RELEASED(cudaMemcpy(d_odata, VPTR(h_odata), VSIZE(h_odata),cudaMemcpyHostToDevice),release);

	auto tmp = h_idata;
	auto start = CLK;
	auto cpu_res = cpuRecursiveReduce(&tmp[0], tmp.size());
	PTIME(start, "cpuRecursiveReduce");
	std::cout << "cpu result:" << cpu_res << std::endl;

	//运行一次避免首次误差
	call_neighbored_reduce(d_idata,d_odata,block,grid);
	CHECK_WITH_RELEASED(cudaDeviceSynchronize(),release);
	CHECK_WITH_RELEASED(cudaMemcpy(d_idata, VPTR(h_idata), VSIZE(h_idata),cudaMemcpyHostToDevice),release);


	start = CLK;
	call_neighbored_reduce(d_idata,d_odata,block,grid);
	CHECK_WITH_RELEASED(cudaDeviceSynchronize(),release);
	PTIME(start, "call_neighbored_reduce");
	CHECK_WITH_RELEASED(cudaMemcpy(VPTR(h_odata), d_odata, VSIZE(h_odata), cudaMemcpyDeviceToHost),release);
	int sum = 0;
	for (auto v:h_odata)
	{
		sum += v;
	}
	std::cout << "neighbored reduce result:" << sum << std::endl;

	CHECK_WITH_RELEASED(cudaMemcpy(d_idata, VPTR(h_idata), VSIZE(h_idata),cudaMemcpyHostToDevice),release);

	start = CLK;
	call_recursive_reduce(d_idata,d_odata,block,grid);
	CHECK_WITH_RELEASED(cudaDeviceSynchronize(),release);
	PTIME(start, "call_neighbored_reduce");

	CHECK_WITH_RELEASED(cudaMemcpy(VPTR(h_odata), d_odata, VSIZE(h_odata), cudaMemcpyDeviceToHost),release);
	sum = 0;
	for (auto v:h_odata)
	{
		sum += v;
	}
	std::cout << "neighbored reduce result:" << sum << std::endl;
}

