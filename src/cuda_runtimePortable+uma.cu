/// runtime portable + uma
#include <cstdio>
#include <thread>
#include <future>
#include "GPUIntegration/utility.h"
#include "matOps_kernels.cu"

void allocate(bool cuda, double*& p, int elemN)
{
	if (cuda){
		cudaMallocManaged(&p, elemN*sizeof(double));
	}else{
		p= new double[elemN];
	}
}

void execute(bool cuda, const int n, const int times, const double* in, double* out)
{
	if(cuda){
		printf("Executing GPU:\n");
		dim3 grid((n-1)/BLOCK_SIZE/BLOCK_SIZE+1);
		dim3 block(BLOCK_SIZE*BLOCK_SIZE);
		longrunning_kernel<<<grid,block>>>(n, times, in, out);
		cudaStreamSynchronize(cudaStreamPerThread);
	}else{
		printf("Executing CPU:\n");
		for(int i=0; i<n; i++){
			out[i]= 0;
	    for(int t=0; t<times; t++){
	      out[i]+= in[i];
	    }
		}
	}
}

void memfree(bool cuda, double* p)
{
	if(cuda){
		cudaFree(p);
	}else{
		delete(p);
	}
}

int main()
{
	/**Checking presence of GPU**/
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  bool cuda= !(error_id == cudaErrorNoDevice || deviceCount == 0);

	double *in, *out;
	long n;
	n= 20;
	std::packaged_task<void()> allocateTask([&] {
		allocate(cuda, in, n);
		allocate(cuda, out, n);
	});
  std::thread(std::move(allocateTask)).join();
  
	for(long i=0; i<n; i++) in[i]= 10*sin(PI/100*i);
	for(long i=0; i<n; i++) out[i]= 1;

	std::packaged_task<void()> executeTask([&] {
		execute(cuda, n, 100, in, out);
	});
  std::future<void> futureHandle = executeTask.get_future();  
  std::thread(std::move(executeTask)).detach();

	futureHandle.get();
	printf("IN:\n");
	for(int i=0; i< n; i++)
		printf("%0.2f\t", in[i]);
	printf("\nOUT:\n");
	for(int i=0; i< n; i++)
		printf("%.0f\t", out[i]);
	printf("\nDONE\n");

	std::packaged_task<void()> memfreeTask([&] {
		memfree(cuda, in);
		memfree(cuda, out);
	});
	std::thread(std::move(memfreeTask)).join();
	return 0;
}
