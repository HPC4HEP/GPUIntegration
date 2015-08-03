#include <iostream>
#include <vector>
#include <chrono>
#include "GPUIntegration/task_service/thread_pool.h"

using namespace edm::service;

int test1();
int test2();

int main()
{
  return test2();
}

int test2(){
  std::cout<<"\n";
  ThreadPoolService pool(std::thread::hardware_concurrency());
  std::future<void> result= pool.enqueue([&] (){
    std::cout<< "Hello task!\n";
  });
  result.get();
  return 0;
}
int test1(){
  std::cout<<"\n";
  ThreadPoolService pool(4);
  std::vector< std::future<int> > results;

  int b;
  for(int i = 0; i < 8; ++i) {
    results.emplace_back(
      pool.enqueue([i] (int a){
        std::cout << "hello " << i << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "world " << i << std::endl;
        return i*i;
      }, b)
    );
  }

  for(auto && result: results)
    std::cout << result.get() << ' ';
  std::cout << std::endl;

  return 0;
}
