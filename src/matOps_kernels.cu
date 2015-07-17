#define BLOCK_SIZE 32

__global__ void matAdd_kernel(int m, int n, const float* __restrict__ A, 
                              const float* __restrict__ B, float* __restrict__ C)
{
	int x= blockIdx.x*blockDim.x + threadIdx.x;
	int y= blockIdx.y*blockDim.y + threadIdx.y;

  // ### Difference between manual and automatic kernel grid:
	//if (x<n && y<m)
  if (y*n+x < n*m)
		C[y*n+x]= A[y*n+x]+B[y*n+x];
}

__global__ void longrunning_kernel(long n, long times, 
                                   const double* in, double* out)
{
  int x= blockIdx.x*blockDim.x + threadIdx.x;

  if (x < n){
    out[x]= 0;
    for(int i=0; i<times; i++){
      out[x]+= in[x];
    }
  }
}

#define CRows ARows
#define CCols BCols
#define BRows ACols
__global__ void matMul_shared_kernel(int ARows, int ACols, int BCols,
                              const float* A, const float* B, float* C)
{
  float CValue = 0; 
  int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
  int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
  for (int k = 0; k < (BLOCK_SIZE + ACols - 1)/BLOCK_SIZE; k++) {
    if (k*BLOCK_SIZE + threadIdx.x < ACols && Row < ARows)
      As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*BLOCK_SIZE + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0;

    if (k*BLOCK_SIZE + threadIdx.y < BRows && Col < BCols)
      Bs[threadIdx.y][threadIdx.x] = B[(k*BLOCK_SIZE + threadIdx.y)*BCols + Col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();
    for (int n = 0; n < BLOCK_SIZE; ++n)
      CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

    __syncthreads();
  }
  if (Row < CRows && Col < CCols)
    C[(blockIdx.y * blockDim.y + threadIdx.y)*CCols+blockIdx.x*blockDim.x+threadIdx.x]= CValue;
}

__global__ void matMul_kernel(int ARows, int ACols, int BCols,
                              const float* A, const float* B, float* C)
{
  float CValue = 0;
  int Row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
  int Col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
  for (int k = 0; k < (BLOCK_SIZE + ACols - 1)/BLOCK_SIZE; k++) {
    for (int n = 0; n < BLOCK_SIZE; ++n)
      if ((k*BLOCK_SIZE + n < ACols && Row < ARows) && (k*BLOCK_SIZE + n < BRows && Col < BCols))
        CValue += A[Row*ACols + k*BLOCK_SIZE + n] * B[(k*BLOCK_SIZE + n)*BCols + Col];
  }
  if (Row < CRows && Col < CCols)
    C[(blockIdx.y * blockDim.y + threadIdx.y)*CCols+blockIdx.x*blockDim.x+threadIdx.x]= CValue;
}
