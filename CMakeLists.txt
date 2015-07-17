### http://bikulov.org/blog/2013/12/24/example-of-cmake-file-for-cuda-plus-cpp-code/
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.10)
project(GPUIntegration)

find_package(CUDA REQUIRED)

INCLUDE_DIRECTORIES(
	include
	../hemi
)

list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3")
list(APPEND CUDA_NVCC_FLAGS
	"--compiler-options -fno-strict-aliasing -lineinfo -Xptxas -dlcm=cg")
list(APPEND CUDA_NVCC_FLAGS "--default-stream per-thread -gencode arch=compute_35,code=sm_35")

CUDA_ADD_EXECUTABLE(cudaMatOpsBasic src/cudaMatOps_basic.cu)
CUDA_ADD_EXECUTABLE(cudaMatOpsConcurrent src/cudaMatOps_concurrent.cu)

############ HEMI ############
CUDA_ADD_EXECUTABLE(cudaMatOpsHEMI src/cudaMatOps_hemi.cu)

############ UMA ############
CUDA_ADD_EXECUTABLE(cudaMatOpsUMA src/cudaMatOps_uma.cu)

############ PortableLayer + UMA ############
CUDA_ADD_EXECUTABLE(cudaMatOpsPUMA src/cudaMatOps_PortableUma.cu)

############ Streams Concurrency Exp ############
CUDA_ADD_EXECUTABLE(streamConcurrency src/stream_ops_concurrent.cu)

############ Raw portability experiment ############
if(DEFINED CUDA_VERSION)
	CUDA_ADD_LIBRARY(cudaMatOpsRawPUMA_lib
									 src/implementations/rawPortable_implementations.cu)
	# target_link_libraries(cudaMatOpsRawPUMA_lib ${CUDA_LIBRARIES})
else(DEFINED CUDA_VERSION)
	add_library(cudaMatOpsRawPUMA_lib
							src/implementations/rawPortable_implementations.cpp)
endif(DEFINED CUDA_VERSION)
add_executable(cudaMatOpsRawPUMA src/cudaMatOps_rawPortable+uma.cpp)
target_link_libraries(cudaMatOpsRawPUMA cudaMatOpsRawPUMA_lib)
