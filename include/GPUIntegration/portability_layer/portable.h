/// CUDA Portability and comfort :) header
/// Based on HEMI
#pragma once

#if defined(__CUDACC__)
	#define PCUDA_COMPILER
	#define PLAUNCHABLE __global__
#else
	#define PHOST_COMPILER
	#define PLAUNCHBLE
#endif


// Automatically include the other sections of the project
#include "launch.h"
