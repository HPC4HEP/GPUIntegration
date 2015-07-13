### http://bikulov.org/blog/2013/12/24/example-of-cmake-file-for-cuda-plus-cpp-code/
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(GPUIntegration)

FIND_PACKAGE(CUDA REQUIRED)

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3")
LIST(APPEND CUDA_NVCC_FLAGS
	"-ccbin $ENV{CXX} --compiler-options -fno-strict-aliasing -lineinfo -Xptxas -dlcm=cg")
LIST(APPEND CUDA_NVCC_FLAGS "--default-stream per-thread -gencode arch=compute_35,code=sm_35")

CUDA_ADD_EXECUTABLE(cudaMatOpsBasic src/cudaMatOps_basic.cu)
CUDA_ADD_EXECUTABLE(cudaMatOpsConcurrent src/cudaMatOps_concurrent.cu)

############ HEMI ############
#INCLUDE_DIRECTORIES(../hemi)
#CUDA_ADD_EXECUTABLE(cudaMatOpsHEMI src/cudaMatOps_hemi.cu)

############ UMA ############
CUDA_ADD_EXECUTABLE(cudaMatOpsUMA src/cudaMatOps_uma.cu)
