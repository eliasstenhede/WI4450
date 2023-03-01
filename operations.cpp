#include "operations.hpp"
#include <omp.h>

void init(int n, double* x, double value)
{
  for (int ix=0; ix<n; ix++) {
    x[ix] = value;
  }
  return;
}

double dot(int n, double const* x, double const* y)
{
  double dot_result = 0.0;
  for (int ix=0; ix<n; ix++) {
    dot_result += x[ix]*y[ix];
  }
  return dot_result;
}

void axpby(int n, double a, double const* x, double b, double* y)
{
  for (int ix=0; ix<n; ix++) {
    y[ix] = a*x[ix]+b*y[ix];
  }
  return;
}

//! apply a 7-point stencil to a vector
void apply_stencil3d(stencil3d const* S, double const* u, double* v)
{
  // Handle the boundaries, where we have dirichlet BC's. 
	// We want to keep those values so just copy u to v, with the correct scaling.
	for (int iz=0; iz<S->nz; iz++) {
		for (int iy=0; iy<S->ny; iy++) {
			v[S->index_c(0      , iy, iz)] =  S->value_c * u[S->index_c(0      , iy, iz)];
			v[S->index_c(S->nx-1, iy, iz)] =  S->value_c * u[S->index_c(S->nx-1, iy, iz)];
		}
		for (int ix=0; ix<S->nx; ix++) {
			v[S->index_c(ix, 0      , iz)] =  S->value_c * u[S->index_c(ix, 0      , iz)];
			v[S->index_c(ix, S->ny-1, iz)] =  S->value_c * u[S->index_c(ix, S->ny-1, iz)];
		}
	}
	for (int ix=1; ix<S->nx-1; ix++) {
		for (int iy=1; iy<S->ny-1; iy++) {
      int bot_idx = S->index_c(ix, iy, 0);
			v[S->index_c(ix, iy, 0      )] =  S->value_c * u[S->index_c(ix, iy, 0      )];
			v[S->index_c(ix, iy, S->nz-1)] =  S->value_c * u[S->index_c(ix, iy, S->nz-1)];
		}
	}
	//For all interior points, calculate the laplacian
	for (int iz=1; iz<S->nz-1; iz++) {
		for (int iy=1; iy<S->ny-1; iy++) {
			for (int ix=1; ix<S->nx-1; ix++) {
        int curr_idx = S->index_c(ix, iy, iz);
				v[curr_idx] =  S->value_c * u[curr_idx];
			  v[curr_idx] += S->value_w * u[S->index_w(ix, iy, iz)];
			  v[curr_idx] += S->value_e * u[S->index_e(ix, iy, iz)];
			  v[curr_idx] += S->value_s * u[S->index_s(ix, iy, iz)];
			  v[curr_idx] += S->value_n * u[S->index_n(ix, iy, iz)];
			  v[curr_idx] += S->value_t * u[S->index_t(ix, iy, iz)];
			  v[curr_idx] += S->value_b * u[S->index_b(ix, iy, iz)];
			}
		}
	}
	return;
}

