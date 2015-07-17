/// portable + uma
#include <cstdio>
#include <thread>
#include <future>
#include "GPUIntegration/utility.h"
#include "GPUIntegration/rawPortable_implementations.h"

int main()
{
	double *in, *out;
	long n;
	n= 20;
	std::packaged_task<void()> allocateTask([&] {
		allocate(in, n);
		allocate(out, n);
	});
  std::thread(std::move(allocateTask)).join();
  
	for(long i=0; i<n; i++) in[i]= 10*sin(PI/100*i);
	for(long i=0; i<n; i++) out[i]= 1;

	std::packaged_task<void()> executeTask([&] {
		execute(n, 100, in, out);
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
		memfree(in);
		memfree(out);
	});
	std::thread(std::move(memfreeTask)).join();
	return 0;
}
