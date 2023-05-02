#include "operations.hpp"
#include "timer.hpp"
#include <omp.h>
#include <string>
//Initialize x with constant value
void init(long n, double* x, double const value)
{
	Timer t("init", 0/1e9, 2*8*n/1e9); //Load and store 
#pragma omp parallel for schedule(static)
  for (int ix=0; ix<n; ix++) {
    x[ix] = value;
  }
  return;
}

double dot(long n, double const* x, double const* y)
{
	Timer t("dot", 2*n/1e9, 2*8*n/1e9); //Counting mult and add as separate operations, we read x and y
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
	Timer t("axpby", 3*n/1e9, 3*8*n/1e9); //Counting multiply and add as separate operations, we load x,y store y.
#pragma omp parallel for schedule(static)
  for (int ix=0; ix<n; ix++) {
    y[ix] = a*x[ix]+b*y[ix];
  }
  return;
}

//Apply a 7-point stencil to a vector
void apply_stencil3d(stencil3d const* S, double const* u, double* v)
{
	long nn = S->nx*S->ny*S->nz;
	Timer tt("separate", 13*nn/1e9, 3*8*nn/1e9); //Load u, load and store v
	//A for loop over the three dimensions that applies the stencil S to vector u and stores it in v

	//    0,    0,    0
	v[S->index_c(0,0,0)] = S->value_c*u[S->index_c(0,0,0)] + S->value_e*u[S->index_e(0,0,0)] + S->value_n*u[S->index_n(0,0,0)] + S->value_t*u[S->index_t(0,0,0)];
	//  ...,    0,    0

	#pragma omp parallel for schedule(static)
	for(int i = 1; i < S->nx-1; i++){
		v[S->index_c(i,0,0)] = S->value_c*u[S->index_c(i,0,0)] + S->value_e*u[S->index_e(i,0,0)] + S->value_w*u[S->index_w(i,0,0)] + S->value_n*u[S->index_n(i,0,0)] + S->value_t*u[S->index_t(i,0,0)];
	}
	// nx-1,    0,    0
	v[S->index_c(S->nx-1,0,0)] = S->value_c*u[S->index_c(S->nx-1,0,0)] + S->value_w*u[S->index_w(S->nx-1,0,0)] + S->value_n*u[S->index_n(S->nx-1,0,0)] + S->value_t*u[S->index_t(S->nx-1,0,0)];

	//    0,  ...,    0
	#pragma omp parallel for schedule(static)
	for(int j = 1; j < S->ny-1; j++){
		v[S->index_c(0,j,0)] = S->value_c*u[S->index_c(0,j,0)] + S->value_e*u[S->index_e(0,j,0)] + S->value_n*u[S->index_n(0,j,0)] + S->value_s*u[S->index_s(0,j,0)] + S->value_t*u[S->index_t(0,j,0)];
	}
	//  ...,  ...,    0
	#pragma omp parallel for schedule(static) collapse(2)
	for(int j = 1; j < S->ny-1; j++){
		for(int i = 1; i < S->nx-1; i++){
			v[S->index_c(i,j,0)] = S->value_c*u[S->index_c(i,j,0)] + S->value_e*u[S->index_e(i,j,0)] + S->value_w*u[S->index_w(i,j,0)] + S->value_n*u[S->index_n(i,j,0)] + S->value_s*u[S->index_s(i,j,0)] + S->value_t*u[S->index_t(i,j,0)];
		}
	}
	// nx-1,  ...,    0
	#pragma omp parallel for schedule(static)
	for(int j = 1; j < S->ny-1; j++){
		v[S->index_c(S->nx-1,j,0)] = S->value_c*u[S->index_c(S->nx-1,j,0)] + S->value_w*u[S->index_w(S->nx-1,j,0)] + S->value_n*u[S->index_n(S->nx-1,j,0)]+ S->value_s*u[S->index_s(S->nx-1,j,0)] + S->value_t*u[S->index_t(S->nx-1,j,0)];
	}

	//    0, ny-1,    0
	v[S->index_c(0,S->ny-1,0)] = S->value_c*u[S->index_c(0,S->ny-1,0)] + S->value_e*u[S->index_e(0,S->ny-1,0)] + S->value_s*u[S->index_s(0,S->ny-1,0)] + S->value_t*u[S->index_t(0,S->ny-1,0)];
	//  ..., ny-1,    0
	#pragma omp parallel for schedule(static)
	for(int i = 1; i < S->nx-1; i++){
		v[S->index_c(i,S->ny-1,0)] = S->value_c*u[S->index_c(i,S->ny-1,0)] + S->value_e*u[S->index_e(i,S->ny-1,0)] + S->value_w*u[S->index_w(i,S->ny-1,0)] + S->value_s*u[S->index_s(i,S->ny-1,0)] + S->value_t*u[S->index_t(i,S->ny-1,0)];
	}
	// nx-1, ny-1,    0
	v[S->index_c(S->nx-1,S->ny-1,0)] = S->value_c*u[S->index_c(S->nx-1,S->ny-1,0)] + S->value_w*u[S->index_w(S->nx-1,S->ny-1,0)] + S->value_s*u[S->index_s(S->nx-1,S->ny-1,0)] + S->value_t*u[S->index_t(S->nx-1,S->ny-1,0)];



	//    0,    0,  ...
	#pragma omp parallel for schedule(static)
	for(int k = 1; k < S->nz-1; k++){
		v[S->index_c(0,0,k)] = S->value_c*u[S->index_c(0,0,k)] + S->value_e*u[S->index_e(0,0,k)] + S->value_n*u[S->index_n(0,0,k)] + S->value_t*u[S->index_t(0,0,k)] + S->value_b*u[S->index_b(0,0,k)];
	}
	//  ...,    0,  ...
	#pragma omp parallel for schedule(static) collapse(2)
	for(int k = 1; k < S->nz-1; k++){
		for(int i = 1; i < S->nx-1; i++){
			v[S->index_c(i,0,k)] = S->value_c*u[S->index_c(i,0,k)] + S->value_e*u[S->index_e(i,0,k)] + S->value_w*u[S->index_w(i,0,k)] + S->value_n*u[S->index_n(i,0,k)] + S->value_t*u[S->index_t(i,0,k)] + S->value_b*u[S->index_b(i,0,k)];
		}
	}
	// nx-1,    0,  ...
	#pragma omp parallel for schedule(static)
	for(int k = 1; k < S->nz-1; k++){
		v[S->index_c(S->nx-1,0,k)] = S->value_c*u[S->index_c(S->nx-1,0,k)] + S->value_w*u[S->index_w(S->nx-1,0,k)] + S->value_n*u[S->index_n(S->nx-1,0,k)] + S->value_t*u[S->index_t(S->nx-1,0,k)] + S->value_b*u[S->index_b(S->nx-1,0,k)];
	}


	//    0,  ...,  ...
	#pragma omp parallel for schedule(static) collapse(2)
	for(int k = 1; k < S->nz-1; k++){
		for(int j = 1; j < S->ny-1; j++){
			v[S->index_c(0,j,k)] = S->value_c*u[S->index_c(0,j,k)] + S->value_e*u[S->index_e(0,j,k)] + S->value_n*u[S->index_n(0,j,k)] + S->value_s*u[S->index_s(0,j,k)] + S->value_t*u[S->index_t(0,j,k)] + S->value_b*u[S->index_b(0,j,k)] ;
		}
	}
	//  ...,  ...,  ...
	#pragma omp parallel for schedule(static) collapse(3)
	for(int k = 1; k < S->nz-1; k++){
		for(int j = 1; j < S->ny-1; j++){
			for(int i = 1; i < S->nx-1; i++){
				v[S->index_c(i,j,k)] = S->value_c*u[S->index_c(i,j,k)] + S->value_e*u[S->index_e(i,j,k)] + S->value_w*u[S->index_w(i,j,k)] + S->value_n*u[S->index_n(i,j,k)] + S->value_s*u[S->index_s(i,j,k)] + S->value_t*u[S->index_t(i,j,k)] + S->value_b*u[S->index_b(i,j,k)];
			}
		}
	}
	// nx-1,  ...,  ...
	#pragma omp parallel for schedule(static) collapse(2)
	for(int k = 1; k < S->nz-1; k++){
		for(int j = 1; j < S->ny-1; j++){
			v[S->index_c(S->nx-1,j,k)] = S->value_c*u[S->index_c(S->nx-1,j,k)] + S->value_w*u[S->index_w(S->nx-1,j,k)] + S->value_n*u[S->index_n(S->nx-1,j,k)]+ S->value_s*u[S->index_s(S->nx-1,j,k)] + S->value_t*u[S->index_t(S->nx-1,j,k)] + S->value_b*u[S->index_b(S->nx-1,j,k)];
		}
	}

	//    0, ny-1,  ...
	#pragma omp parallel for schedule(static)
	for(int k = 1; k < S->nz-1; k++){
		v[S->index_c(0,S->ny-1,k)] = S->value_c*u[S->index_c(0,S->ny-1,k)] + S->value_e*u[S->index_e(0,S->ny-1,k)] + S->value_s*u[S->index_s(0,S->ny-1,k)] + S->value_t*u[S->index_t(0,S->ny-1,k)] + S->value_b*u[S->index_b(0,S->ny-1,k)];
	}  
	//  ..., ny-1,  ...
	#pragma omp parallel for schedule(static) collapse(2)
	for(int k = 1; k < S->nz-1; k++){
		for(int i = 1; i < S->nx-1; i++){
			v[S->index_c(i,S->ny-1,k)] = S->value_c*u[S->index_c(i,S->ny-1,k)] + S->value_e*u[S->index_e(i,S->ny-1,k)] + S->value_w*u[S->index_w(i,S->ny-1,k)] + S->value_s*u[S->index_s(i,S->ny-1,k)] + S->value_t*u[S->index_t(i,S->ny-1,k)] + S->value_b*u[S->index_b(i,S->ny-1,k)] ;
		}
	}
	// nx-1, ny-1,  ...
	#pragma omp parallel for schedule(static)
	for(int k = 1; k < S->nz-1; k++){
		v[S->index_c(S->nx-1,S->ny-1,k)] = S->value_c*u[S->index_c(S->nx-1,S->ny-1,k)] + S->value_w*u[S->index_w(S->nx-1,S->ny-1,k)] + S->value_s*u[S->index_s(S->nx-1,S->ny-1,k)] + S->value_t*u[S->index_t(S->nx-1,S->ny-1,k)] + S->value_b*u[S->index_b(S->nx-1,S->ny-1,k)];
	}


	//    0,    0, nz-1
	v[S->index_c(0,0,S->nz-1)] = S->value_c*u[S->index_c(0,0,S->nz-1)] + S->value_e*u[S->index_e(0,0,S->nz-1)] + S->value_n*u[S->index_n(0,0,S->nz-1)] + S->value_b*u[S->index_b(0,0,S->nz-1)];
	//  ...,    0, nz-1
	#pragma omp parallel for schedule(static)
	for(int i = 1; i < S->nx-1; i++){
		v[S->index_c(i,0,S->nz-1)] = S->value_c*u[S->index_c(i,0,S->nz-1)] + S->value_e*u[S->index_e(i,0,S->nz-1)] + S->value_w*u[S->index_w(i,0,S->nz-1)] + S->value_n*u[S->index_n(i,0,S->nz-1)] + S->value_b*u[S->index_b(i,0,S->nz-1)];
	}
	// nx-1,    0, nz-1
	v[S->index_c(S->nx-1,0,S->nz-1)] = S->value_c*u[S->index_c(S->nx-1,0,S->nz-1)] + S->value_w*u[S->index_w(S->nx-1,0,S->nz-1)] + S->value_n*u[S->index_n(S->nx-1,0,S->nz-1)] + S->value_b*u[S->index_b(S->nx-1,0,S->nz-1)];

	//    0,  ..., nz-1
	#pragma omp parallel for schedule(static)
	for(int j = 1; j < S->ny-1; j++){
		v[S->index_c(0,j,S->nz-1)] = S->value_c*u[S->index_c(0,j,S->nz-1)] + S->value_e*u[S->index_e(0,j,S->nz-1)] + S->value_n*u[S->index_n(0,j,S->nz-1)] + S->value_s*u[S->index_s(0,j,S->nz-1)] + S->value_b*u[S->index_b(0,j,S->nz-1)];
	}  
	//  ...,  ..., nz-1
	#pragma omp parallel for schedule(static) collapse(2)
	for(int j = 1; j < S->ny-1; j++){
		for(int i = 1; i < S->nx-1; i++){
			v[S->index_c(i,j,S->nz-1)] = S->value_c*u[S->index_c(i,j,S->nz-1)] + S->value_e*u[S->index_e(i,j,S->nz-1)] + S->value_w*u[S->index_w(i,j,S->nz-1)] + S->value_n*u[S->index_n(i,j,S->nz-1)] + S->value_s*u[S->index_s(i,j,S->nz-1)] + S->value_b*u[S->index_b(i,j,S->nz-1)];
		}
	}
	// nx-1,  ..., nz-1
	#pragma omp parallel for schedule(static)
	for(int j = 1; j < S->ny-1; j++){
		v[S->index_c(S->nx-1,j,S->nz-1)] = S->value_c*u[S->index_c(S->nx-1,j,S->nz-1)] + S->value_w*u[S->index_w(S->nx-1,j,S->nz-1)] + S->value_n*u[S->index_n(S->nx-1,j,S->nz-1)]+ S->value_s*u[S->index_s(S->nx-1,j,S->nz-1)] + S->value_b*u[S->index_b(S->nx-1,j,S->nz-1)];
	}

	//    0, ny-1, nz-1
	v[S->index_c(0,S->ny-1,S->nz-1)] = S->value_c*u[S->index_c(0,S->ny-1,S->nz-1)] + S->value_e*u[S->index_e(0,S->ny-1,S->nz-1)] + S->value_s*u[S->index_s(0,S->ny-1,S->nz-1)] + S->value_b*u[S->index_b(0,S->ny-1,S->nz-1)];
	//  ..., ny-1, nz-1
	#pragma omp parallel for schedule(static)
	for(int i = 1; i < S->nx-1; i++){
	v[S->index_c(i,S->ny-1,S->nz-1)] = S->value_c*u[S->index_c(i,S->ny-1,S->nz-1)] + S->value_e*u[S->index_e(i,S->ny-1,S->nz-1)] + S->value_w*u[S->index_w(i,S->ny-1,S->nz-1)] + S->value_s*u[S->index_s(i,S->ny-1,S->nz-1)] + S->value_b*u[S->index_b(i,S->ny-1,S->nz-1)];
	}
	// nx-1, ny-1, nz-1
	v[S->index_c(S->nx-1,S->ny-1,S->nz-1)] = S->value_c*u[S->index_c(S->nx-1,S->ny-1,S->nz-1)] + S->value_w*u[S->index_w(S->nx-1,S->ny-1,S->nz-1)] + S->value_s*u[S->index_s(S->nx-1,S->ny-1,S->nz-1)] + S->value_b*u[S->index_b(S->nx-1,S->ny-1,S->nz-1)];

	return;
}

