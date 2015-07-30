#include <thread>
#include <functional>
#include <memory>
#include <future>
#include <unordered_map>

class TaskServiceStat{
		class TaskInterface{};
		template<typename Fn>
		class TaskWrapper: public TaskInterface{
				std::packaged_task<Fn> task_;
				std::thread thread_;
			public:
				TaskWrapper(std::function<Fn>&& f):
										task_(std::forward< std::function<Fn> >(f)) {};
				template<typename... Args>
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
		template<typename Fn, typename... Args>
		std::future<void> launch(int ID, Args&&... args){
			return static_cast<TaskWrapper<Fn>*>(tasks_.at(ID).get())->
																					 launch(std::forward<Args>(args)...);
		}

	private:
		std::unordered_map<int, TaskInterfacePtr> tasks_;
};
