### http://bikulov.org/blog/2013/12/24/example-of-cmake-file-for-cuda-plus-cpp-code/
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.7)
project(GPUIntegration)

find_package(CUDA)

include_directories(
	include
	../hemi
)

list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -pthread")#-Wl --no-as-needed")
list(APPEND CUDA_NVCC_FLAGS
	"--compiler-options -fno-strict-aliasing -lineinfo -Xptxas -dlcm=cg")
list(APPEND CUDA_NVCC_FLAGS "--default-stream per-thread -gencode arch=compute_35,code=sm_35")

################################################################################
if(CUDA_FOUND)

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

############ Runtime portability ############
	CUDA_ADD_EXECUTABLE(runtimePortable src/cuda_runtimePortable+uma.cu)
	CUDA_ADD_EXECUTABLE(runtimePortableDynamicPol src/cuda_runtimePortable_dynamicPol.cu)

############ Service ############
	#CUDA_ADD_EXECUTABLE(serviceStatic src/service_staticPolym.cu)
	add_executable(testService src/service_test.cpp)
	set_target_properties(testService
                        PROPERTIES COMPILE_FLAGS "-std=c++14 -O3 -pthread")


endif(CUDA_FOUND)

############ Configure-time portability ############
if(CUDA_FOUND)
	message(Using\ GPU)
	CUDA_ADD_LIBRARY(cudaMatOpsRawPUMA_lib
									 src/implementations/rawPortable_implementations.cu)
	# target_link_libraries(cudaMatOpsRawPUMA_lib ${CUDA_LIBRARIES})
else(CUDA_FOUND)
	message(Using\ CPU)
	add_library(cudaMatOpsRawPUMA_lib
							src/implementations/rawPortable_implementations.cpp)
endif(CUDA_FOUND)

add_executable(cudaMatOpsRawPUMA src/cudaMatOps_rawPortable+uma.cpp)
target_link_libraries(cudaMatOpsRawPUMA cudaMatOpsRawPUMA_lib)
