CXX=g++
CXX_FLAGS=-O3 -march=native -g -fopenmp -std=c++17

DEFS=-DNDEBUG

#default target (built when typing just "make")
default: run_tests.x main_cg_poisson.x cg_solver.x

# general rule to comple a C++ source file into an object file
%.o: %.cpp
	${CXX} -c ${CXX_FLAGS} ${DEFS} $<

#define some dependencies on headers
operations.o: operations.hpp timer.hpp
cg_solver.o: cg_solver.hpp operations.hpp timer.hpp
cg_poisson.o: cg_solver.hpp operations.hpp timer.hpp
gtest_mpi.o: gtest_mpi.hpp
benchmark_stencil.o: timer.hpp operations.hpp

TEST_SOURCES=test_operations.cpp test_cg_solver.cpp
MAIN_OBJ=main_cg_poisson.o cg_solver.o operations.o timer.o

run_tests.x: run_tests.cpp ${TEST_SOURCES} gtest_mpi.o operations.o timer.o
	${CXX} ${CXX_FLAGS} ${DEFS} -o run_tests.x $^

main_cg_poisson.x: ${MAIN_OBJ}
	${CXX} ${CXX_FLAGS} ${DEFS} -o main_cg_poisson.x $^

cg_solver.x: ${MAIN_OBJ}
	${CXX} ${CXX_FLAGS} ${DEFS} -o cg_solver.x $^

stencil.x: benchmark_stencil.cpp timer.o
	${CXX} ${CXX_FLAGS} ${DEFS} -o stencil.x $^


test: run_tests.x
	./run_tests.x

stencil: stencil.x

clean:
	-rm *.o *.x

# phony targets are run regardless of dependencies being up-to-date
PHONY: clean, test

