#include <cstdio>
#include <thread>
#include <future>
#include <functional>
#include <memory>
#include <map>
#include "GPUIntegration/utility.h"
#include "matOps_kernels.cu"


class TaskService{
		class TaskInterface{
			public:
				virtual ~TaskInterface() {};
				virtual std::future<void> launch() =0;
		};
		template<typename Fn> class TaskWrapper;
		template<typename R, typename... Args>
		class TaskWrapper<R(Args...)>: public TaskInterface{
				std::packaged_task<R(Args...)> task_;
				std::thread thread_;
			public:
				TaskWrapper(std::function<R(Args...)>&& f):
										task_(std::forward< std::function<R(Args...)> >(f)) {};
				std::future<void> launch(Args&&... args){
					std::future<void> future= task_.get_future();
					thread_= std::thread(std::move(task_), std::forward<Args>(args)...);
					thread_.detach();
					return future;
				}
		};
		typedef std::unique_ptr<TaskInterface> TaskInterfacePtr;

	public:
		template<typename Fn>
		void set_task(int ID, std::function<Fn>&& f){
			tasks_[ID]= std::move(TaskInterfacePtr(
			          		new TaskWrapper<Fn>(std::forward< std::function<Fn> >(f)) ));
		}
		template<typename... Args>
		std::future<void> launch(int ID, Args&&... args){
			return tasks_.at(ID)->launch(std::forward<Args>(args)...);
		}

	private:
		std::map<int, TaskInterfacePtr> tasks_;
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
  TaskService taskService;
  Implementation *impl;
	/**Checking presence of GPU**/
  int deviceCount= 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if(error_id == cudaSuccess && deviceCount > 0) impl= new GPU;
  else impl= new CPU;
  /*if (error_id == cudaErrorNoDevice || deviceCount == 0) impl= new CPU;
  else impl= new GPU;*/
  //impl= new CPU;

	double *in, *out;
	long n;
	n= 20;
	taskService.set_task<void()>(0, [&] {
		impl->allocate(in, n);
		impl->allocate(out, n);
	});
  taskService.launch(0).get();
  
	for(long i=0; i<n; i++) in[i]= 10*sin(PI/100*i);
	for(long i=0; i<n; i++) out[i]= 1;

	taskService.set_task<void()>(1, [&] {
		impl->execute(n, 100, in, out);
	});
  std::future<void> future1= taskService.launch(1);

	future1.get();
	printf("IN:\n");
	for(int i=0; i< n; i++)
		printf("%0.2f\t", in[i]);
	printf("\nOUT:\n");
	for(int i=0; i< n; i++)
		printf("%.0f\t", out[i]);
	printf("\nDONE\n");

	taskService.set_task<void()>(2, [&] {
		impl->memfree(in);
		impl->memfree(out);
	});
	taskService.launch(2).get();
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
