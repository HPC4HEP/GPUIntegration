#pragma once

//To be templated

inline void allocate(double*& p, int elemN);
inline void execute(const int n, const int times, const double* in, double* out);
inline void memfree(double* p);
