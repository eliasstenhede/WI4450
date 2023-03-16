#include "gtest_mpi.hpp"
#include "cg_solver.cpp"
#include <iostream>
#include <cmath>

double f(double x, double y, double z)
{
  return sin(2*M_PI*x)*sin(2*M_PI*y)*sin(2*M_PI*z);
}

stencil3d laplace3d_stencil(int nx, int ny, int nz)
{
  if (nx<=2 || ny<=2 || nz<=2) throw std::runtime_error("need at least two grid points in each direction to implement boundary conditions.");
  stencil3d L;
  L.nx=nx; L.ny=ny; L.nz=nz;
  double dx=1.0/(nx-1), dy=1.0/(ny-1), dz=1.0/(nz-1);
  L.value_c = (2.0/(dx*dx) + 2.0/(dy*dy) + 2.0/(dz*dz));
  L.value_n = -1.0/(dy*dy);
  L.value_e = -1.0/(dx*dx);
  L.value_s = -1.0/(dy*dy);
  L.value_w = -1.0/(dx*dx);
  L.value_t = -1.0/(dz*dz);
  L.value_b = -1.0/(dz*dz);
  return L;
}
//Checks that the solution is symmetric given symmetric BC
TEST(cg_solver, check_sol_symmetric)
{
	const int nx=7;
	const int ny=nx;
	const int nz=ny;
	const int n=nx*ny*nz;
	const double init_val = 1.7;
  double dx=1.0/(nx-1), dy=1.0/(ny-1), dz=1.0/(nz-1);
  
  // Laplace operator
  stencil3d L = laplace3d_stencil(nx,ny,nz);

  // solution vector: start with a 0 vector
  double *x = new double[n];
  init(n, x, 0.0);

  // right-hand side
  double *b = new double[n];
  init(n, b, 0.0);

  // Dirichlet boundary conditions at z=0 (others are 0 in our case, initialized above)
  for (int j=0; j<ny; j++)
    for (int i=0; i<nx; i++)
    {
      b[L.index_c(0, i, j)] -= i*j*(1.0-i*dx)*(1.0-j*dx);
      b[L.index_c(i, 0, j)] -= i*j*(1.0-i*dx)*(1.0-j*dx);
      b[L.index_c(i, j, 0)] -= i*j*(1.0-i*dx)*(1.0-j*dx);
    }

  // solve the linear system of equations using CG
  int numIter, maxIter=100;
  double resNorm, tol=std::sqrt(std::numeric_limits<double>::epsilon());

	cg_solver(&L, n, x, b, tol, maxIter, &resNorm, &numIter, 0);
	//Check symmetry by looping through some aribtrary axes
  for (int ix=0; ix<nx; ix++) {
	  EXPECT_NEAR(x[L.index_c(ix, 3, ix)], x[L.index_c(3, ix, ix)], 10.0*std::sqrt(std::numeric_limits<double>::epsilon()));
	  EXPECT_NEAR(x[L.index_c(ix, 1, ix)], x[L.index_c(1, ix, ix)], 10.0*std::sqrt(std::numeric_limits<double>::epsilon()));
	  EXPECT_NEAR(x[L.index_c(1 , 1, ix)], x[L.index_c(1, ix, 1 )], 10.0*std::sqrt(std::numeric_limits<double>::epsilon()));
  }
	delete [] x;
  delete [] b;

}

//Checks that x converges to the analytic solution for homogenous dirichlet boundary conditions and a sine-like forcing term.
TEST(cg_solver, check_sol_sine)
{
	const int nx=40;
	const int ny=40;
	const int nz=40;
	const int n=nx*ny*nz;
  double dx=1.0/(nx-1), dy=1.0/(ny-1), dz=1.0/(nz-1);
  
  // Laplace operator
  stencil3d L = laplace3d_stencil(nx,ny,nz);

  // solution vector: start with a 0 vector
  double *x = new double[n];
  init(n, x, 0.0);

  // right-hand side
  double *b = new double[n];
  init(n, b, 0.0);

  for (int k=0; k<nz; k++)
  {
    for (int j=0; j<ny; j++)
    {
      for (int i=0; i<nx; i++)
      {
        int idx = L.index_c(i,j,k);
        b[idx] = f(i*dx,j*dy,k*dz);
      }
    }
  }

  // solve the linear system of equations using CG
  int numIter, maxIter=500;
  double resNorm, tol=std::sqrt(std::numeric_limits<double>::epsilon());

  cg_solver(&L, n, x, b, tol, maxIter, &resNorm, &numIter, 0);
  
	float max_err = 0.0;
	for (int k=1; k<nz-1; k++) {
    for (int j=1; j<ny-1; j++) {
      for (int i=1; i<nx-1; i++) {
        int idx = L.index_c(i,j,k);
				//analytic solution for homogenous dirichlet and sine forcing function
				float expected = 1.0/(4*3*M_PI*M_PI)*f(i*dx,j*dy,k*dz);
				float err = x[L.index_c(i,j,k)] - expected;
				if (err < 0.0)
					err *= -1.0;
				if (err > max_err)
					max_err = err;
			}
    }
  }
	EXPECT_NEAR(max_err, 0, 0.001);
	delete [] x;
  delete [] b;

}
