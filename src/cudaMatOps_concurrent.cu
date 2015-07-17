/// Concurrent version
#include <cstdio>
#include <thread>
#include <future>
#include "GPUIntegration/utility.h"
#include "matOps_kernels.cu"

int main()
{
	float *A, *B, *C;
	// m: number of rows
	// n: number of columns
	int m, n;
	// Simple testcase
	m= 10; n= m;
	A= new float[m*n], B= new float[m*n], C= new float[m*n];
	init(A, B, m*n);


	std::packaged_task<void()> task([&]{
		float *dA, *dB, *dC;
		cudaMalloc((void **) &dA, m*n*sizeof(float));
		cudaMalloc((void **) &dB, m*n*sizeof(float));
		cudaMalloc((void **) &dC, m*n*sizeof(float));
		cudaMemcpy(dA, A, m*n*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dB, B, m*n*sizeof(float), cudaMemcpyHostToDevice);
		dim3 grid((n-1)/BLOCK_SIZE+1, (m-1)/BLOCK_SIZE+1);
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		matAdd_kernel<<<grid,block>>>(m, n, dA, dB, dC);
		//(m==n)? matMul_kernel<<<grid,block>>>(m,m,m, dA, dB, dC): exit(1);
		cudaMemcpy(C, dC, m*n*sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(dA); cudaFree(dB); cudaFree(dC);
	});
  std::future<void> futureKernel = task.get_future();  
  std::thread(std::move(task)).detach();


	//Output
	printf("A:\n");
	show(A, m,n);
	printf("\nB:\n");
	show(B, m,n);

	// Wait for kernel to complete and show result
  futureKernel.get();
	printf("\nC:\n");
	show(C, m,n);
	printf("Done\n\n");
	free(A); free(B); free(C);
	return 0;
}
