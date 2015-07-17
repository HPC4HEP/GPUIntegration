#pragma once

#include "execution_policy.h"
#include "error_checking.h"
#include "configure.h"

namespace portable{

// Automatic launch functions for __global__ kernel function pointers: CUDA only
template <typename... Arguments>
void launch(void(*f)(Arguments... args), Arguments... args)
{
	#ifdef PCUDA_COMPILER
		ExecutionPolicy p;
		launch(p, f, args...);
	#else
		f(args...);
	#endif
}

// Launch __global__ kernel function with explicit configuration
template <typename... Arguments>
#ifdef PCUDA_COMPILER
	void launch(const ExecutionPolicy &policy, void (*f)(Arguments...), Arguments... args)
	{
		ExecutionPolicy p = policy;
		checkCuda(configureGrid(p, f));
		f<<<p.getGridSize(), p.getBlockSize(), p.getSharedMemBytes()>>>(args...);
	}
#else
	void launch(const ExecutionPolicy&, void (*f)(Arguments...), Arguments... args)
	{
		f(args...);
	}
#endif

/*	VERSION 2
// Automatic Launch functions for closures (functor or lambda)
template <typename Function, typename... Arguments>
void launch(Function f, Arguments... args)
{
	#ifdef HEMI_CUDA_COMPILER
		ExecutionPolicy p;
		launch(p, f, args...);
	#else
		Kernel(f, args...);
	#endif
}

// Launch with explicit (or partial) configuration
template <typename Function, typename... Arguments>
#ifdef HEMI_CUDA_COMPILER
	void launch(const ExecutionPolicy &policy, Function f, Arguments... args)
	{
		ExecutionPolicy p = policy;
		checkCuda(configureGrid(p, Kernel<Function, Arguments...>));
		Kernel<<<p.getGridSize(), p.getBlockSize(), p.getSharedMemBytes()>>>(f, args...);
	}
#else
	void launch(const ExecutionPolicy&, Function f, Arguments... args)
	{
		Kernel(f, args...);
	}
#endif
*/
} // namespace portable
