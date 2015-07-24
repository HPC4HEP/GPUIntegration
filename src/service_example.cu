/// runtime portable + uma
#include <cstdio>
#include <thread>
#include <future>
#include <functional>
#include <memory>
#include <map>
#include "GPUIntegration/utility.h"
#include "matOps_kernels.cu"

class TaskService{
		typedef std::packaged_task<void()> VoidTask;
		typedef std::unique_ptr<VoidTask>  VoidTaskUniqPtr;
	public:
		void set_task(int ID, std::function<void()>&& f){
			printf("Progress Check#%d\n",2);
			tasks[ID]= std::move(VoidTaskUniqPtr(new VoidTask(std::move(f))));
			/*tasks.insert(std::pair<int, VoidTaskUniqPtr>(ID,
			             std::move( VoidTaskUniqPtr(new VoidTask(std::move(f))) )));*/
			printf("Progress Check#%d\n",3);
		}
		std::future<void> launch(int ID){
			std::future<void> future= tasks.at(ID)->get_future();
			std::thread(std::move(*tasks.at(ID))).detach();
			tasks.erase(ID);
			return future;
		}
	private:
		std::map<int, VoidTaskUniqPtr> tasks;
};

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
  TaskService *taskService;
  Implementation *impl;
	/**Checking presence of GPU**/
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id == cudaErrorNoDevice || deviceCount == 0) impl= new CPU;
  else impl= new GPU;

  printf("Progress Check#%d\n",1);
	double *in, *out;
	long n;
	n= 20;
	// set_task<>(int ID, Function f)
	taskService->set_task(0, [&] {
		impl->allocate(in, n);
		impl->allocate(out, n);
	});
  taskService->launch(0).get();
  
	for(long i=0; i<n; i++) in[i]= 10*sin(PI/100*i);
	for(long i=0; i<n; i++) out[i]= 1;

	taskService->set_task(1, [&] {
		impl->execute(n, 100, in, out);
	});
  std::future<void> future1= taskService->launch(1);

	future1.get();
	printf("IN:\n");
	for(int i=0; i< n; i++)
		printf("%0.2f\t", in[i]);
	printf("\nOUT:\n");
	for(int i=0; i< n; i++)
		printf("%.0f\t", out[i]);
	printf("\nDONE\n");

	taskService->set_task(2, [&] {
		impl->memfree(in);
		impl->memfree(out);
	});
	taskService->launch(2).get();
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
