#include "../matOps_kernels.cu"

void allocate(double*& p, int elemN)
{
	cudaMallocManaged(&p, elemN*sizeof(double));// cudaMemAttachHost
}

void execute(const int n, const int times, const double* in, double* out)
{
	dim3 grid((n-1)/BLOCK_SIZE/BLOCK_SIZE+1);
	dim3 block(BLOCK_SIZE*BLOCK_SIZE);
	longrunning_kernel<<<grid,block>>>(n, times, in, out);
	cudaStreamSynchronize(cudaStreamPerThread);
}

void memfree(double* p)
{
	cudaFree(p);
}
