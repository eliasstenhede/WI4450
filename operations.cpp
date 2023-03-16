#include "operations.hpp"
#include "timer.hpp"
#include <omp.h>

//Initialize x with constant value
void init(long n, double* x, double value)
{
	Timer t("init", 0/1e9, n*8/1e9);
#pragma omp for schedule(static)
  for (int ix=0; ix<n; ix++) {
    x[ix] = value;
  }
  return;
}

double dot(long n, double const* x, double const* y)
{
	Timer t("dot", 2*n/1e9, 8*2*n/1e9); //Assuming no fused multiply add, we read x and y
  double dot_result = 0.0;
#pragma omp parallel for schedule(static) reduction(+:dot_result)
  for (int ix=0; ix<n; ix++) {
    dot_result += x[ix]*y[ix];
  }
  return dot_result;
}

//y = a*x+b*y
void axpby(long n, double a, double const* x, double b, double* y)
{
	Timer t("axpby", 3*n/1e9, 8*3*n/1e9); //Assuming no fused multiply add, we read x, y and write to y
#pragma omp parallel for schedule(static)
  for (int ix=0; ix<n; ix++) {
    y[ix] = a*x[ix]+b*y[ix];
  }
  return;
}

//Apply a 7-point stencil to a vector
void apply_stencil3d(stencil3d const* S, double const* u, double* v)
{
	long nx = S->nx;
	long ny = S->ny;
	long nz = S->nz;
	long n = nx*ny*nz;
	Timer t("stencil", (6*n+7*n)/1e9, 2*n/1e9); //For all 7 loops we perform multiplication, for last 6 we add. We assume u and v are cached so we only read each of them once.
#pragma omp parallel
	{
#pragma omp for schedule(static) 
  	for (int iz=0; iz<nz; iz++) {
  	  for (int iy=0; iy<ny; iy++) {
  	    for (int ix=0; ix<nx; ix++) {
          v[S->index_c(ix, iy, iz)] = S->value_c * u[S->index_c(ix, iy, iz)];
  		  }
  		}
  	}
#pragma omp for schedule(static)   
  	for (int iz=0; iz<nz; iz++) {
  	  for (int iy=0; iy<ny; iy++) {
  	    for (int ix=0; ix<nx-1; ix++) {
          v[S->index_c(ix, iy, iz)] += S->value_e * u[S->index_e(ix, iy, iz)];
  		  }
  		}
  	}
#pragma omp for schedule(static)   
  	for (int iz=0; iz<nz; iz++) {
  	  for (int iy=0; iy<ny; iy++) {
  	    for (int ix=1; ix<nx; ix++) {
          v[S->index_c(ix, iy, iz)] += S->value_w * u[S->index_w(ix, iy, iz)];
  		  }
  		}
  	}
#pragma omp for schedule(static)   
  	for (int iz=0; iz<nz; iz++) {
  	  for (int iy=0; iy<ny-1; iy++) {
  	    for (int ix=0; ix<nx; ix++) {
          v[S->index_c(ix, iy, iz)] += S->value_n * u[S->index_n(ix, iy, iz)];
  		  }
  		}
  	}
#pragma omp for schedule(static)   
  	for (int iz=0; iz<nz; iz++) {
  	  for (int iy=1; iy<ny; iy++) {
  	    for (int ix=0; ix<nx; ix++) {
          v[S->index_c(ix, iy, iz)] += S->value_s * u[S->index_s(ix, iy, iz)];
  		  }
  		}
  	}
#pragma omp for schedule(static)   
  	for (int iz=0; iz<nz-1; iz++) {
  	  for (int iy=0; iy<ny; iy++) {
  	    for (int ix=0; ix<nx; ix++) {
          v[S->index_c(ix, iy, iz)] += S->value_t * u[S->index_t(ix, iy, iz)];
  		  }
  		}
  	}
#pragma omp for schedule(static)   
  	for (int iz=1; iz<nz; iz++) {
  	  for (int iy=0; iy<ny; iy++) {
  	    for (int ix=0; ix<nx; ix++) {
          v[S->index_c(ix, iy, iz)] += S->value_b * u[S->index_b(ix, iy, iz)];
  		  }
  		}
  	}
	}
	return;
}

