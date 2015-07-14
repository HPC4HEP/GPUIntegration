#include <cstdio>
#include <cmath>
#define PI 3.14159265

void init(float *A, float *B, int size)
{
	for (int i=0; i<size; i++)
		A[i]= 10*sin(PI/100*i), B[i]= sin(PI/100*i+PI/6)*sin(PI/100*i+PI/6);
}

void show(const float *mat, int m, int n)
{
	for (int y = 0; y < m; y++){
		for (int x = 0; x < n; x++){
			printf("%.2f\t",y,x, mat[y*n+x]);
		}
		printf("\n");
	}
}
