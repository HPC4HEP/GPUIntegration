/// Goal: no boilerplate
#include <cstdio>
#include <cmath>
#include <thread>
#include <future>
#include "matOps_kernels.cu"

#define PI 3.14159265

int main()
{
	float *A, *B, *C;
	// m: number of rows
	// n: number of columns
	int m, n;
	// Simple testcase
	m= 10; n= m;
	A= new float[m*n], B= new float[m*n], C= new float[m*n];
	for (int i=0; i<m*n; i++)
		A[i]= 10*sin(PI/100*i), B[i]= sin(PI/100*i+PI/6)*sin(PI/100*i+PI/6);


	std::packaged_task<void()> task([&](){
		kernel_run(matAdd_kernel, grid, m, n, dA, dB, dC);
	});
  std::future<void> futureKernel = task.get_future();  
  std::thread(std::move(task)).detach();


	//Output
	printf("A:\n");
	for (int y = 0; y < m; y++){
		for (int x = 0; x < n; x++){
			printf("%.2f\t",y,x, A[y*n+x]);
		}
		printf("\n");
	}
	printf("\nB:\n");
	for (int y = 0; y < m; y++){
		for (int x = 0; x < n; x++){
			printf("%.2f\t",y,x, B[y*n+x]);
		}
		printf("\n");
	}

	// Wait for kernel to complete and show result
  futureKernel.get();
	printf("\nC:\n");
	for (int y = 0; y < m; y++){
		for (int x = 0; x < n; x++){
			printf("%.2f\t",y,x, C[y*n+x]);
		}
		printf("\n");
	}
	free(A); free(B); free(C);
	return 0;
}
