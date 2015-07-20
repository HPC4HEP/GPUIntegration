/// runtime portable + uma
#include <cstdio>
#include <thread>
#include <future>
#include "GPUIntegration/utility.h"
#include "matOps_kernels.cu"

class Implementation{
	public:
		virtual void allocate(double*& p, int elemN) =0;
		virtual void execute(const int n, const int times, const double* in, double* out) =0;
		virtual void memfree(double* p) =0;
};
class GPU: public Implementation{
	public:
		void allocate(double*& p, int elemN);
		void execute(const int n, const int times, const double* in, double* out);
		void memfree(double* p);
};
class CPU: public Implementation{
	public:
		void allocate(double*& p, int elemN);
		void execute(const int n, const int times, const double* in, double* out);
		void memfree(double* p);
};

int main()
{
  Implementation *dev;
  GPU __gpu;
  CPU __cpu;
	/**Checking presence of GPU**/
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id == cudaErrorNoDevice || deviceCount == 0) dev= &__cpu;
  else dev= &__gpu;

	double *in, *out;
	long n;
	n= 20;
	std::packaged_task<void()> allocateTask([&] {
		dev->allocate(in, n);
		dev->allocate(out, n);
	});
  std::thread(std::move(allocateTask)).join();
  
	for(long i=0; i<n; i++) in[i]= 10*sin(PI/100*i);
	for(long i=0; i<n; i++) out[i]= 1;

	std::packaged_task<void()> executeTask([&] {
		dev->execute(n, 100, in, out);
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
		dev->memfree(in);
		dev->memfree(out);
	});
	std::thread(std::move(memfreeTask)).join();
	return 0;
}

void CPU::allocate(double*& p, int elemN)
{
	p= new double[elemN];
}

void CPU::execute(const int n, const int times, const double* in, double* out)
{
	printf("Executing CPU:\n");
	for(int i=0; i<n; i++){
		out[i]= 0;
    for(int t=0; t<times; t++){
      out[i]+= in[i];
    }
	}
}

void CPU::memfree(double* p)
{
	delete(p);
}

void GPU::allocate(double*& p, int elemN)
{
	cudaMallocManaged(&p, elemN*sizeof(double));// cudaMemAttachHost
}

void GPU::execute(const int n, const int times, const double* in, double* out)
{
	printf("Executing GPU:\n");
	dim3 grid((n-1)/BLOCK_SIZE/BLOCK_SIZE+1);
	dim3 block(BLOCK_SIZE*BLOCK_SIZE);
	longrunning_kernel<<<grid,block>>>(n, times, in, out);
	cudaStreamSynchronize(cudaStreamPerThread);
}

void GPU::memfree(double* p)
{
	cudaFree(p);
}
