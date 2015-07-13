/// UMA version
#include <cstdio>
#include <cmath>
#include <thread>
#include <future>
#include "matOps_kernels.cu"

#define PI 3.14159265

void init(float *A, float *B, int size)
{
	for (int i=0; i<size; i++)
		A[i]= 10*sin(PI/100*i), B[i]= sin(PI/100*i+PI/6)*sin(PI/100*i+PI/6);
}

void show(const float *mat, int m, int n)
{
	for (int y = 0; y < m; y++){
		for (int x = 0; x < n; x++){
			printf("%.2f\t",y,x, mat[y*n+x]);
		}
		printf("\n");
	}
}

int main()
{
	float *A, *B, *C;
	// m: number of rows
	// n: number of columns
	int m, n;
	// Simple testcase
	m= 10; n= m;
	//A= new float[m*n], B= new float[m*n], C= new float[m*n];
	cudaMallocManaged((void **)&A, m*n);
	cudaMallocManaged((void **)&B, m*n);
	cudaMallocManaged((void **)&C, m*n);
	init(A, B, m*n);


	std::packaged_task<void()> task([&]{
		/*cudaStreamAttachMemAsync(cudaStreamPerThread, &A);
		cudaStreamAttachMemAsync(cudaStreamPerThread, &B);
		cudaStreamAttachMemAsync(cudaStreamPerThread, &C);*/
		dim3 grid((n-1)/BLOCK_SIZE+1, (m-1)/BLOCK_SIZE+1);
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		matAdd_kernel<<<grid,block>>>(m, n, A, B, C);
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
