#include <iostream>
#include <vector>
#include <future>
#include <mutex>
#include <condition_variable>
#include "GPUIntegration/task_service/thread_pool.h"
#include "../src/matOps_kernels.cu"

using namespace edm::service;

std::mutex mtx;
std::condition_variable cv;
bool ready = false;
long sum= 0;

void print_id (int id) {
  std::unique_lock<std::mutex> lck(mtx);
  while (!ready) cv.wait(lck);
  // ...
  std::cout << id << "\t";
  sum+= id;
}

void go() {
  std::unique_lock<std::mutex> lck(mtx);
  ready = true;
  cv.notify_all();
}

int main (int argc, char** argv)
{
  ThreadPoolService pool;
  std::vector<std::future<void>> futures;
  const int N= (argc>1)? std::stol(argv[1]): 30;
  float *in, *din;
  int n= 2000;
  in= new float[n];
  for(int i=0; i<n; i++) in[i]= 10*cos(3.141592/100*i);
  // Make GPU input data available for all threads
  cudaMalloc((void **) &din, n*sizeof(float));
  cudaMemcpy(din, in, n*sizeof(float), cudaMemcpyHostToDevice);
  if (N>n){
    std::cout<< "Smaller N\n";
    return 1;
  }

  // spawn N threads
  for (int i=0; i<N; ++i){
    // Schedule a CPU or a GPU task
    if (i%5) futures.emplace_back(pool.enqueue(print_id,i+1));
    else futures.emplace_back(pool.enqueue([=] (const int times){
      float *dout;
      cudaMalloc((void **) &dout, n*sizeof(float));
      dim3 grid((n-1)/BLOCK_SIZE/BLOCK_SIZE+1);
      dim3 block(BLOCK_SIZE*BLOCK_SIZE);
      longrunning_FL<<<grid,block>>>(n, times, din, dout);
      cudaStreamSynchronize(cudaStreamPerThread);
      float out;
      cudaMemcpy(&out, dout+i, 1*sizeof(float), cudaMemcpyDeviceToHost);
      std::cout << "GPU::" << out << "\t";
      cudaFree(dout);
    }, 2));
  }
  std::cout << N<< " threads ready to race...\n";
  go();                       // go!

  for (auto& future: futures) future.get();
  std::cout << "\nDONE, sum= "<<sum<<"\n";
	for(int i=0; i<N; i++)
		sum-= (i%5)? i+1: 0;
	if (sum!= 0) std::cout<< "ERROR!\n";
  return 0;
}
