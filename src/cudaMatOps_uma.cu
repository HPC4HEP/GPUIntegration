/// UMA version
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
	//A= new float[m*n], B= new float[m*n], C= new float[m*n];
	cudaMallocManaged(&A, m*n*sizeof(float));
	cudaMallocManaged(&B, m*n*sizeof(float));
	cudaMallocManaged(&C, m*n*sizeof(float));
	init(A, B, m*n);


	std::packaged_task<void()> task([&]{
		/*cudaStreamAttachMemAsync(cudaStreamPerThread, &A);
		cudaStreamAttachMemAsync(cudaStreamPerThread, &B);
		cudaStreamAttachMemAsync(cudaStreamPerThread, &C);*/
		dim3 grid((n-1)/BLOCK_SIZE+1, (m-1)/BLOCK_SIZE+1);
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		matAdd_kernel<<<grid,block>>>(m, n, const_cast<const float*>(A), 
		                              const_cast<const float*>(B), C);
		//(m==n)? matMul_kernel<<<grid,block>>>(m,m,m, dA, dB, dC): exit(1);
		cudaStreamSynchronize(cudaStreamPerThread);
	});
  std::future<void> futureKernel = task.get_future();  
  std::thread(std::move(task)).detach();

  //##### BUG #####
  //Accessing A and B might cause "Bus error"
	//Output
	printf("A:\n");
	show(A, m,n);
	printf("\nB:\n");
	show(B, m,n);

	// Wait for kernel to complete TO REUSE same data
  futureKernel.get();
	printf("\nC:\n");
	show(C, m,n);
	printf("Done\n\n");
	//free(A); free(B); free(C);
	cudaFree(A); cudaFree(B); cudaFree(C);
	return 0;
}
