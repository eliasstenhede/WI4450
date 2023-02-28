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
  for (int ix=0; ix<S->nx; ix++) {
		for (int iy=0; iy<S->ny; iy++) {
			for (int iz=0; iz<S->nz; iz++) {
        int curr_idx = S->index_c(ix, iy, iz);
				v[curr_idx]  = S->value_c * u[curr_idx];
				if (ix > 0)
					v[curr_idx] += S->value_w * u[S->index_w(ix, iy, iz)];
				if (ix+1 < S->nx)
					v[curr_idx] += S->value_e * u[S->index_e(ix, iy, iz)];
				if (iy > 0)
					v[curr_idx] += S->value_s * u[S->index_s(ix, iy, iz)];
				if (iy+1 < S->ny)
					v[curr_idx] += S->value_n * u[S->index_n(ix, iy, iz)];
				if (iz > 0)
					v[curr_idx] += S->value_t * u[S->index_t(ix, iy, iz)];
				if (iz+1 < S->nz)
					v[curr_idx] += S->value_b * u[S->index_b(ix, iy, iz)];
			}
		}
	}
	return;
}

