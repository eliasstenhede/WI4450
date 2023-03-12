#include "operations.hpp"
#include "cg_solver.hpp"
#include "timer.hpp"

#include <iostream>
#include <cmath>
#include <limits>

#include <cmath>

// Forcing term
double f(double x, double y, double z)
{
  return z*sin(2*M_PI*x)*std::sin(M_PI*y) + 8*z*z*z;
}

// boundary condition at z=0
double g_0(double x, double y)
{
  return x*(1.0-x)*y*(1-y);
}

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

int run(int nx, int ny, int nz)
{
  // total number of unknowns
  int n=nx*ny*nz;

  double dx=1.0/(nx-1), dy=1.0/(ny-1), dz=1.0/(nz-1);

  // Laplace operator
  stencil3d L = laplace3d_stencil(nx,ny,nz);

  // solution vector: start with a 0 vector
  double *x = new double[n];
  init(n, x, 0.0);

  // right-hand side
  double *b = new double[n];
  init(n, b, 0.0);

  // initialize the rhs with f(x,y,z) in the interior of the domain
#pragma omp parallel for schedule(static)
  for (int k=1; k<nz-1; k++)
  {
    double z = k*dz;
    for (int j=1; j<ny-1; j++)
    {
      double y = j*dy;
      for (int i=1; i<nx-1; i++)
      {
        double x = i*dx;
        int idx = L.index_c(i,j,k);
        b[idx] = f(x,y,z);
      }
    }
  }
  // Dirichlet boundary conditions at z=0 (others are 0 in our case, initialized above)
  for (int j=0; j<ny; j++)
    for (int i=0; i<nx; i++)
    {
      b[L.index_c(i,j,0)] -= g_0(i*dx, j*dy)/(dz*dz);
    }

  // solve the linear system of equations using CG
  int numIter, maxIter=10;
  double resNorm, tol=std::sqrt(std::numeric_limits<double>::epsilon());

  try {
    cg_solver(&L, n, x, b, tol, maxIter, &resNorm, &numIter, 0);
  } catch(std::exception e)
  {
    std::cerr << "Caught an exception in cg_solve: " << e.what() << std::endl;
    exit(-1);
  }
	printf("Iterations before convergence for grid size (%d, %d, %d): %d.\n", nx, ny, nz, numIter);
  delete [] x;
  delete [] b;

  return 0;
}

int main(int argc, char* argv[]) {
  int nx, ny, nz;

  if      (argc==1) {nx=128;           ny=128;           nz=128;}
  else if (argc==2) {nx=atoi(argv[1]); ny=nx;            nz=nx;}
  else if (argc==4) {nx=atoi(argv[1]); ny=atoi(argv[2]); nz=atoi(argv[3]);}
  else {std::cerr << "Invalid number of arguments (should be 0, 1 or 3)"<<std::endl; exit(-1);}
  if (ny<0) ny=nx;
  if (nz<0) nz=nx;
	for (int ix=0; ix<1; ix++) {
	  {
			Timer t("Total");
		  run(nx, ny, nz);
	  }
	}
	
  Timer::summarize();
	return 0;
}
