/**
Copyright (c) 2012 Jakob Progsch, Václav Zeman
This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:
   1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
   2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
   3. This notice may not be removed or altered from any source
   distribution.

--> This is an altered version of the original code.
Editor: Konstantinos Samaras-Tsakiris, kisamara@auth.gr
*/

#ifndef FWCore_Services_TaskService_h
#define FWCore_Services_TaskService_h

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <memory>
#include <functional>
#include <stdexcept>

/*
//CMSSW Integration
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
*/

namespace edm{
namespace service{

// std::thread pool for resources recycling
class ThreadPoolService {
public:
  // the constructor just launches some amount of workers
  //ThreadPoolService(const edm::ParameterSet&, edm::ActivityRegistry&)
	ThreadPoolService(size_t threads_n = std::thread::hardware_concurrency()) : stop_(false)
	{
		if(!threads_n)
			throw std::invalid_argument("more than zero threads expected");

		this->workers_.reserve(threads_n);
		for(; threads_n; --threads_n)
			this->workers_.emplace_back([this] (){
	     	while(true)
	     	{
	     		std::function<void()> task;

	     		{
	     			std::unique_lock<std::mutex> lock(this->queue_mutex_);
	     			this->condition_.wait(lock,
	     			                     [this]{ return this->stop_ || !this->tasks_.empty(); });
	     			if(this->stop_ && this->tasks_.empty())
	     				return;
	     			task = std::move(this->tasks_.front());
	     			this->tasks_.pop();
	     		}

	     		task();
	     }
      });
	}
  // deleted copy&move ctors&assignments
	ThreadPoolService(const ThreadPoolService&) = delete;
	ThreadPoolService& operator=(const ThreadPoolService&) = delete;
	ThreadPoolService(ThreadPoolService&&) = delete;
	ThreadPoolService& operator=(ThreadPoolService&&) = delete;

  // add new work item to the pool
  template<class F, class... Args>
	std::future<typename std::result_of<F(Args...)>::type> enqueue(F&& f, Args&&... args)
	{
		using packaged_task_t = std::packaged_task<typename std::result_of<F(Args...)>::type ()>;

		std::shared_ptr<packaged_task_t> task(new packaged_task_t(
                      std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    ));
		auto resultFut = task->get_future();
		{
			std::unique_lock<std::mutex> lock(this->queue_mutex_);
			this->tasks_.emplace([task](){ (*task)(); });
		}
		this->condition_.notify_one();
		return resultFut;
	}

  // the destructor joins all threads
	virtual ~ThreadPoolService()
	{
		this->stop_ = true;
		this->condition_.notify_all();
		for(std::thread& worker : this->workers_)
			worker.join();
	}
private:
  // need to keep track of threads so we can join them
	std::vector< std::thread > workers_;
  // the task queue
	std::queue< std::function<void()> > tasks_;

  // synchronization
	std::mutex queue_mutex_;
	std::condition_variable condition_;
  // workers_ finalization flag
	std::atomic_bool stop_;
};

}	// namespace service
}	// namespace edm

//DEFINE_FWK_SERVICE(ZombieKillerService);
#endif FWCore_Services_TaskService_h
