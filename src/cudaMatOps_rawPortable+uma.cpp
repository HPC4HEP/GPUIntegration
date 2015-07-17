/// portable + uma
#include <cstdio>
#include <thread>
#include <future>
#include "GPUIntegration/rawPortable_implementations.h"

int main()
{
	double *in, *out;
	long n;
	n= 10;
	
	std::packaged_task<void()> allocate([&] {
		allocate(in, n);
		allocate(out, n);
	});
  std::thread(std::move(allocate)).join();
	for(long i=0; i<n; i++) in[i]= 10*sin(PI/100*i);
	for(long i=0; i<n; i++) out[i]= 1;

	std::packaged_task<void()> execute([&] {
		execute();
	});
  std::future<void> futureHandle = execute.get_future();  
  std::thread(std::move(execute)).detach();

	futureHandle.get();
	printf("IN:\n");
	/*
	for(int i=0; i< n; i++)
		printf("%0.2f\t", in[i]);
	printf("\nOUT:\n");
	for(int i=0; i< n; i++)
		printf("%.0f\t", out[i]);
	*/
	printf("\nDONE\n");

	std::packaged_task<void()> memfree([&] {
		memfree(in);
		memfree(out);
	});
	std::thread(std::move(memfree)).join();
	return 0;
}
