/// P(-ortable)UMA version
#include <cstdio>
#include <thread>
#include <future>
#include "utility.h"
#include "matOps_kernels.cu"

#include "GPUIntegration/portable.h"

int main()
{
	float *A, *B, *C;
	// m: number of rows
	// n: number of columns
	int m, n;
	// Simple testcase
	m= 10; n= m;
	//A= new float[m*n], B= new float[m*n], C= new float[m*n];
	cudaMallocManaged(&A, m*n);
	cudaMallocManaged(&B, m*n);
	cudaMallocManaged(&C, m*n);
	init(A, B, m*n);


	std::packaged_task<void()> task([&]{
		/*cudaStreamAttachMemAsync(cudaStreamPerThread, &A);
		cudaStreamAttachMemAsync(cudaStreamPerThread, &B);
		cudaStreamAttachMemAsync(cudaStreamPerThread, &C);*/
		
		portable::launch(matAdd_kernel, m,n, const_cast<const float*>(A),
		                 const_cast<const float*>(B),C);
		//(m==n)? matMul_kernel<<<grid,block>>>(m,m,m, dA, dB, dC): exit(1);
		cudaStreamSynchronize(cudaStreamPerThread);
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
	//free(A); free(B); free(C);
	cudaFree(A); cudaFree(B); cudaFree(C);
	return 0;
}
