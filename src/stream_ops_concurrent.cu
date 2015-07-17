/// Stream concurrency test
#include <cstdio>
#include <thread>
#include <future>
#include "GPUIntegration/utility.h"
#include "matOps_kernels.cu"

#include "GPUIntegration/portability_layer/portable.h"

int main()
{
	double *in, *out;
	long n;
	n= 20;
	cudaMallocManaged(&in, n*sizeof(double));// cudaMemAttachHost
	cudaMallocManaged(&out, n*sizeof(double));// cudaMemAttachHost
	for(long i=0; i<n; i++) in[i]= 10*sin(PI/100*i);
	for(long i=0; i<n; i++) out[i]= 1;


	// Launch many task threads -- each one operates on a different CUDA stream
	int taskN, totalTasks= 2;
	long partSize= n/totalTasks+1;	//ceiling
	std::future<void> futureHandle[totalTasks];
	for(taskN= 0; taskN< totalTasks; taskN++){
		if (taskN*partSize> n) break;
		std::packaged_task<void()> task([=, &in, &out] {
			double *taskIn;
			double *taskOut;
			taskIn=  in + taskN*partSize;
			taskOut= out+ taskN*partSize;
			long endIdx= ((taskN+1)*partSize<= n)? partSize: n-taskN*partSize;
			/**
			Currently impossible to attach chunks of managed memory to specific streams
			According to the cudaStreamAttachMemAsync API doc
			--> cudaStreamAttachMemAsync(cudaStreamPerThread, &taskIn);
			--> cudaStreamAttachMemAsync(cudaStreamPerThread, &taskOut);
			**/
			portable::launch(longrunning_kernel, endIdx, 100l, const_cast<const double*>(taskIn), taskOut);
			cudaStreamSynchronize(cudaStreamPerThread);
		});
	  futureHandle[taskN] = task.get_future();  
	  std::thread(std::move(task)).detach();
	}


  totalTasks= taskN;
	for(taskN= 0; taskN< totalTasks; taskN++)
		futureHandle[taskN].get();
	printf("IN:\n");
	for(int i=0; i< n; i++)
		printf("%0.2f\t", in[i]);
	printf("\nOUT:\n");
	for(int i=0; i< n; i++)
		printf("%.0f\t", out[i]);
	printf("\nDONE\n");
	cudaFree(in); cudaFree(out);
	return 0;
}
