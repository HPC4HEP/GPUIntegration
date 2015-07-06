### http://bikulov.org/blog/2013/12/24/example-of-cmake-file-for-cuda-plus-cpp-code/
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(GPUIntegration)

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O2 ")
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -Xptxas -dlcm=cg)
# LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
# LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)

CUDA_ADD_EXECUTABLE(cudaMatOps src/cudaMatOps.cu)
CUDA_ADD_EXECUTABLE(cudaMatOpsConcurrent src/cudaMatOps_concurrent.cu)

############		Probably unnecessary		#############
#INCLUDE_DIRECTORIES(/usr/local/cuda/include)
#TARGET_LINK_LIBRARIES(cudaMatOps /usr/local/cuda/lib64/libcudart.so)
