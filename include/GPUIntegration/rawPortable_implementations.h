#pragma once

//To be templated

void allocate(double*& p, int elemN);
void execute(const int n, const int times, const double* in, double* out);
void memfree(double* p);
