void allocate(double*& p, int elemN)
{
	p= new double[elemN];
}

void execute(const int n, const int times, const double* in, double* out)
{
	for(int i=0; i<n; i++){
		out[i]= 0;
    for(int t=0; t<times; t++){
      out[i]+= in[i];
    }
	}
}

void memfree(double* p)
{
	delete(p);
}
