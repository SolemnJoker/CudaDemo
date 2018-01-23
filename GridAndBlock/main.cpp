#include "cudaheader.h"
#include "commen.h"
#include <vector>
void initData(std::vector<float>& arr)
{
	for (auto& v:arr)
	{
		v = (rand()&0xFF)/10.0f;
	}
}
int main()
{
	const int nx = 1 << 12;
	const int ny = 1 << 12;
    const int arraySize = 1<<24;
	const int bytesSize = arraySize * sizeof(float);
    std::vector<float> a(arraySize);
    std::vector<float> b(arraySize);
    std::vector<float> c(arraySize,0.0f);
	initData(a);
	initData(b);
	float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
	auto release = [&]() {
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_c);
	};

	CHECK(cudaSetDevice(0));
	CHECK_WITH_RELEASED(cudaMalloc(&dev_a, bytesSize), release);
	CHECK_WITH_RELEASED(cudaMalloc(&dev_b, bytesSize), release);
	CHECK_WITH_RELEASED(cudaMalloc(&dev_c, bytesSize), release);

	CHECK_WITH_RELEASED(cudaMemcpy(dev_a, &a[0], bytesSize,cudaMemcpyHostToDevice), release);
	CHECK_WITH_RELEASED(cudaMemcpy(dev_b, &b[0], bytesSize,cudaMemcpyHostToDevice), release);
	CHECK_WITH_RELEASED(cudaMemcpy(dev_c, &c[0], bytesSize,cudaMemcpyHostToDevice), release);

	//
	addWithCuda(dev_a, dev_b, dev_c, nx, ny);

	CHECK_WITH_RELEASED(cudaGetLastError(),release);

	CHECK_WITH_RELEASED(cudaDeviceSynchronize(), release);
	CHECK_WITH_RELEASED(cudaMemcpy(&c[0], dev_c, bytesSize,cudaMemcpyDeviceToHost), release);

	release();
    CHECK(cudaDeviceReset());
    return 0;
}
