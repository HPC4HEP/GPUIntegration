#include <cstdio>
#include <cmath>
#define PI 3.14159265

inline void init(float *A, float *B, int size)
{
	for (int i=0; i<size; i++)
		A[i]= 10*sin(PI/100*i), B[i]= sin(PI/100*i+PI/6)*sin(PI/100*i+PI/6);
}

inline void show(const float *mat, int m, int n)
{
	for (int y = 0; y < m; y++){
		for (int x = 0; x < n; x++){
			printf("%.2f\t",y,x, mat[y*n+x]);
		}
		printf("\n");
	}
}


//A nice template exercise
//#include <type_traits>
template<class T>
inline void mallocMany(int elemN, T& p)
{
	typename std::remove_pointer<T>::type Traw;
	p= (T)malloc(elemN*sizeof(Traw));
}
template<class T, class... More>
inline void mallocMany(int elemN, T& p, More&... morePs)
{
	typename std::remove_pointer<T>::type Traw;
	p= (T)malloc(elemN*sizeof(Traw));
	mallocMany(elemN, morePs...);
}
