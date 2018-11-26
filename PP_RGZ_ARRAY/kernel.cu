#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <chrono>

cudaError_t addWithCuda(int *c, const int *a, unsigned int size);

using namespace std;

/*__device__ bool Prime(long long n)
{
	for (int i = 2; i <= sqrt((double)n); i++)
		if (n%i == 0)
			return false;
	return true;
}*/

__global__ void addKernel(char *output, long long from, int *a, int cudaCores)
{
	long long current = threadIdx.x + from + cudaCores * blockIdx.x;

	long long outPos = current - from;

	output[outPos] = 0;

	if (a[current] % current == 0) output[outPos] = 1;
	else output[outPos] = -1;
}

int main()
{
	const int arraySize = 10000000;
	int *a = new int[arraySize];
	int *c = new int[arraySize];

	for (int c = 1; c < arraySize; c++)
	{
		a[c - 1] = c;
	}

	// Add vectors in parallel.
	auto begin = chrono::high_resolution_clock::now();
	cudaError_t cudaStatus = addWithCuda(c, a, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	auto end = chrono::high_resolution_clock::now();

	/*int i = 0;
	while (c[i] > 0)
	{
		cout << c[i] << endl;
		i++;
	}*/

	cout << "Work time: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;
	system("pause");

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, unsigned int size)
{
	int cudaCores = 1000;

	long long from = 2;
	const long long bufferSize = size - from;
	const long long blockCount = (bufferSize / cudaCores) + (bufferSize%cudaCores == 0 ? 0 : 1);

	if (bufferSize < cudaCores)
	{
		cudaCores = bufferSize;
	}

	char *output = new char[bufferSize];
	char *dev_output;

	int *dev_a = 0;
	//int *dev_c = 0;
	cudaError_t cudaStatus;
	cudaEvent_t start;
	cudaEvent_t stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_output, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//int threadsPerBlock = 55;
	//int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	// Launch a kernel on the GPU with one thread for each element.

	cudaEventRecord(start, 0);

	addKernel << < blockCount, cudaCores >> > (dev_output, from, dev_a, cudaCores);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaEventRecord(stop, 0);
	float time = 0;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_output, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_output);
	cudaFree(dev_a);

	/*int l = 0;
	while (l < bufferSize)
	{
		cout << c[l] << endl;
		l++;
	}*/

	/*int y = 0;
	while (y < bufferSize)
	{
		if ((int)output[y] > 0) { cout << "Chislo " << output[y] << " ne prostoe" << endl; break; }
		y++;
	}*/

	cout << "Work time: " << time << endl;
	system("pause");

	return cudaStatus;
}