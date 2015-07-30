#include <thread>
#include <functional>
#include <memory>
#include <future>
#include <map>

class TaskServiceVirt{
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
