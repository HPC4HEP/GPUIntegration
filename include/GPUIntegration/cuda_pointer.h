//19/8, 17:50

template<typename T>
class cudaPointer{
public:
	//flag: cudaMemAttachGlobal | cudaMemAttachHost
	cudaPointer(int elementN=1, unsigned flag=cudaMemAttachGlobal): uniqPtr_(new T*){
	#ifdef __NVCC__
		cudaMallocManaged(&uniqPtr_.get(), elementN*sizeof(T), flag);
	#endif
	}
	//uniqPtr_ must retain ownership until here!
	~cudaPointer(){
	#ifdef __NVCC__
		cudaFree(uniqPtr_.get());
	#endif
	}
	//Act on enclosed unique_ptr
	std::unique_ptr<T>& operator()() const{
		return uniqPtr_;
	}
	T* passKernel() const{
		return uniqPtr_.get();
	}
private:
	std::unique_ptr<T> uniqPtr_;
};
