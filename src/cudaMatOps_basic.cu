///Simple CUDA version of matrix addition / Notice the boilerplate
#include <cstdio>
#include <cmath>
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

	// CUDA boilerplate begin {
	float *dA, *dB, *dC;
	cudaMalloc((void **) &dA, m*n*sizeof(float));
	cudaMalloc((void **) &dB, m*n*sizeof(float));
	cudaMalloc((void **) &dC, m*n*sizeof(float));
	cudaMemcpy(dA, A, m*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, m*n*sizeof(float), cudaMemcpyHostToDevice);
	dim3 grid((n-1)/BLOCK_SIZE+1, (m-1)/BLOCK_SIZE+1, 1);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
	matAdd_kernel<<<grid,block>>>(m, n, dA, dB, dC);
	//(m==n)? matMul_kernel<<<grid,block>>>(m,m,m, dA, dB, dC): exit(1);

	cudaMemcpy(C, dC, m*n*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dA); cudaFree(dB); cudaFree(dC);
	// CUDA boilerplate end		}

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
