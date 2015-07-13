/// HEMI implementation
#include <cstdio>
#include <cmath>
#include <thread>
#include <future>
#include "matOps_kernels.cu"

#include "hemi/array.h"
#include "hemi/launch.h"

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
	// m: number of rows
	// n: number of columns
	int m, n;
	// Simple testcase
	m= 10; n= m;

	hemi::Array<float> A(m*n, true), B(m*n, true), C(m*n, true);
	init(A.writeOnlyHostPtr(), B.writeOnlyHostPtr(), m*n);


	std::packaged_task<void()> task([&] (){
		//hemi::ExecutionPolicy policy(ceil(m/BLOCK_SIZE)*ceil(n/BLOCK_SIZE), BLOCK_SIZE);
		hemi::cudaLaunch(matAdd_kernel, m,n, A.readOnlyDevicePtr(), 
		                 B.readOnlyDevicePtr(), C.writeOnlyDevicePtr());
	});
  std::future<void> futureKernel = task.get_future();  
  std::thread(std::move(task)).detach();


	//Output
	/*printf("A:\n");
	show(A.readOnlyHostPtr(), m,n);
	printf("\nB:\n");
	show(B.readOnlyHostPtr(), m,n);
*/
	// Wait for kernel to complete and show result
  futureKernel.get();
	printf("\nC:\n");
	show(C.readOnlyHostPtr(), m,n);
	printf("Done\n\n");
	return 0;
}
