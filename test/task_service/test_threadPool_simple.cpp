#include <iostream>
#include <vector>
#include <future>
#include <mutex>
#include <condition_variable>
#include "GPUIntegration/task_service/thread_pool.h"

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

  // spawn N threads:
  for (int i=0; i<N; ++i)
    futures.emplace_back(pool.enqueue(print_id,i+1));
  std::cout << N<< " threads ready to race...\n";
  go();                       // go!

  for (auto& future: futures) future.get();
  std::cout << "\nDONE, sum= "<<sum<<"\n";
	for(int i=0; i<N; i++)
		sum-= i+1;
	if (sum!= 0) std::cout<< "ERROR!\n";
  return 0;
}
