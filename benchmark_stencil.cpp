#include "operations.hpp"
#include "timer.hpp"
#include <omp.h>
#include <string>
#include <time.h>

stencil3d laplace3d_stencil(int nx, int ny, int nz)
{
	if (nx<=2 || ny<=2 || nz<=2) throw std::runtime_error("need at least two grid points in each direction to implement boundary conditions.");
	stencil3d L;
	L.nx=nx; L.ny=ny; L.nz=nz;
	double dx=1.0/(nx-1), dy=1.0/(ny-1), dz=1.0/(nz-1);
	L.value_c =  2.0/(dx*dx) + 2.0/(dy*dy) + 2.0/(dz*dz);
	L.value_n = -1.0/(dy*dy);
	L.value_e = -1.0/(dx*dx);
	L.value_s = -1.0/(dy*dy);
	L.value_w = -1.0/(dx*dx);
	L.value_t = -1.0/(dz*dz);
	L.value_b = -1.0/(dz*dz);
	return L;
}

void apply_stencil3d_bench_1(stencil3d const* S, double const* u, double* v, int bszy)
{
	long nx = S->nx;
	long ny = S->ny;
	long nz = S->nz;
	long nn = nx*ny*nz;
	Timer tt("original", 13*nn/1e9, 3*8*nn/1e9); //Load u, load and store v
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

void apply_stencil3d_bench_2(stencil3d const* S, double const* u, double* v, int bszy)
{
	long nx = S->nx;
	long ny = S->ny;
	long nz = S->nz;
	long nn = nx*ny*nz;
	Timer tt("Padded", 13*nn/1e9, 3*8*nn/1e9); //Load u, load and store v

#pragma omp parallel for schedule(static)
	for (int iz=1; iz<nz-1; iz++) {
		for (int iy=1; iy<ny-1; iy++) {
			for (int ix=1; ix<nx-1; ix++) {
				double accum = S->value_c * u[S->index_c(ix, iy, iz)];
				accum += S->value_b * u[S->index_b(ix, iy, iz)];
				accum += S->value_t * u[S->index_t(ix, iy, iz)];
				accum += S->value_s * u[S->index_s(ix, iy, iz)];
				accum += S->value_n * u[S->index_n(ix, iy, iz)];
				accum += S->value_w * u[S->index_w(ix, iy, iz)];
				accum += S->value_e * u[S->index_e(ix, iy, iz)];
				v[S->index_c(ix, iy, iz)] = accum;
			}
		}
	}
	return;
}

void apply_stencil3d_bench_3(stencil3d const* S, double const* u, double* v, int bszy)
{
	long nx = S->nx;
	long ny = S->ny;
	long nz = S->nz;
	long nn = nx*ny*nz;
	Timer tt("Padded collapse", 13*nn/1e9, 3*8*nn/1e9); //Load u, load and store v

#pragma omp parallel for schedule(static) collapse(3)
	for (int iz=1; iz<nz-1; iz++) {
		for (int iy=1; iy<ny-1; iy++) {
			for (int ix=1; ix<nx-1; ix++) {
				double accum = S->value_c * u[S->index_c(ix, iy, iz)];
				accum += S->value_b * u[S->index_b(ix, iy, iz)];
				accum += S->value_t * u[S->index_t(ix, iy, iz)];
				accum += S->value_s * u[S->index_s(ix, iy, iz)];
				accum += S->value_n * u[S->index_n(ix, iy, iz)];
				accum += S->value_w * u[S->index_w(ix, iy, iz)];
				accum += S->value_e * u[S->index_e(ix, iy, iz)];
				v[S->index_c(ix, iy, iz)] = accum;
			}
		}
	}
	return;
}

void apply_stencil3d_bench_4(stencil3d const* S, double const* u, double* v, int bszy)
{
	char buffer[50]; // create a character buffer to store the formatted string
	std::snprintf(buffer, sizeof(buffer), "Blocked %d", bszy); // use snprintf to format the string
	std::string str(buffer);

	long nx = S->nx;
	long ny = S->ny;
	long nz = S->nz;
	long nn = nx*ny*nz;
	Timer tt(str, 13*nn/1e9, 3*8*nn/1e9); //Load u, load and store v

	for (int bky=1; bky<ny; bky+=bszy) {
#pragma omp parallel for schedule(static)
		for (int iz=1; iz<nz-1; iz++) {
			int limy = (ny<bszy+bky) ? ny-1 : bszy+bky;
			for (int iy=bky; iy<limy; iy++) {
				for (int ix=1; ix<nx-1; ix++) {
					double accum = S->value_c * u[S->index_c(ix, iy, iz)];
					accum += S->value_b * u[S->index_b(ix, iy, iz)];
					accum += S->value_t * u[S->index_t(ix, iy, iz)];
					accum += S->value_s * u[S->index_s(ix, iy, iz)];
					accum += S->value_n * u[S->index_n(ix, iy, iz)];
					accum += S->value_w * u[S->index_w(ix, iy, iz)];
					accum += S->value_e * u[S->index_e(ix, iy, iz)];
					v[S->index_c(ix, iy, iz)] = accum;
				}
			}
		}
	}
	return;
}

void apply_stencil3d_bench_5(stencil3d const* S, double const* u, double* v, int bszy)
{
	long nx = S->nx;
	long ny = S->ny;
	long nz = S->nz;
	long nn = nx*ny*nz;
	Timer tt("If", 13*nn/1e9, 3*8*nn/1e9); //Load u, load and store v
	
#pragma omp parallel for schedule(static)
	for (int iz=0; iz<nz; iz++) {
		for (int iy=0; iy<ny; iy++) {
			for (int ix=0; ix<nx; ix++) {
				double accum = S->value_c * u[S->index_c(ix, iy, iz)];
				if (iz>0)
     	 			accum += S->value_b * u[S->index_b(ix, iy, iz)];
				if (iz<nz-1)
     	 			accum += S->value_t * u[S->index_t(ix, iy, iz)];
				if (iy>0)
     	 			accum += S->value_s * u[S->index_s(ix, iy, iz)];
				if (iy<ny-1)
     	 			accum += S->value_n * u[S->index_n(ix, iy, iz)];
				if (ix>0)
     	 			accum += S->value_w * u[S->index_w(ix, iy, iz)];
				if (ix<nx-1)
     	 			accum += S->value_e * u[S->index_e(ix, iy, iz)];
				v[S->index_c(ix, iy, iz)] = accum;
			}
		}
	}
	return;
}

void apply_stencil3d_bench_6(stencil3d const* S, double const* u, double* v, int bszy)
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

int main(int argc, char* argv[]) {
	const int n = 600+2;
	int bsz;
	if (argc==2) {
		bsz = atoi(argv[1]);
	} else {
		bsz = 30;
	}
	stencil3d L = laplace3d_stencil(n, n, n);
	double *x = new double[n*n*n];
	double *r = new double[n*n*n];
	
	void (*functions[6])(stencil3d const*, double const*, double*, int) = {
		apply_stencil3d_bench_1,
		apply_stencil3d_bench_2,
		apply_stencil3d_bench_3,
		apply_stencil3d_bench_4,
		apply_stencil3d_bench_5,
		apply_stencil3d_bench_6
	};

	srand(time(NULL));
	int iters = 700;
	for(int it=0; it<iters; it++) {
		(*functions[rand()%6])(&L, x, r, bsz);
	}
	delete [] x;
	delete [] r;
	Timer::summarize();
}
