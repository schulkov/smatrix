FC = gfortran
FFLAGS = -O3 -Wall -std=f2008
CC = gcc
CFLAGS = -O3 -Wall
NVCC = nvcc
NVCFLAGS = -O3 --shared -arch=sm_70 --compiler-options -Wall
LIBS = -lcudart

# Avoid funny character set dependencies
unexport LC_ALL
LC_COLLATE=C
LC_NUMERIC=C
export LC_COLLATE LC_NUMERIC

# Avoid interference with shell env settings
unexport GREP_OPTIONS

PHONY := make clean
make: smatrix_sgf_dense.x smatrix_sgf_sparse.x prescreen_smatrix_sgf_sparse.x

kinds.o: kinds.f90
	$(FC) $(FFLAGS) -c -o $@ $<

mathconstants.o: mathconstants.f90 kinds.o
	$(FC) $(FFLAGS) -c -o $@ $<

mathlib.o: mathlib.f90 mathconstants.o
	$(FC) $(FFLAGS) -c -o $@ $<

orbtramat.o: orbtramat.f90 mathconstants.o mathlib.o
	$(FC) $(FFLAGS) -c -o $@ $<

ai_overlap.o: ai_overlap.cu mathconstants.o orbtramat.o
	$(NVCC) $(NVCFLAGS) -c -o $@ $< $(LIBS)

cgf_utils.o: cgf_utils.f90 kinds.o orbtramat.o ai_overlap.o
	$(FC) $(FFLAGS) -c -o $@ $<

smatrix_sgf_dense.x: smatrix_sgf_dense.f90 kinds.o ai_overlap.o mathconstants.o orbtramat.o mathlib.o cgf_utils.o
	$(FC) $(FFLAGS) -o $@ $^ $(LIBS)

smatrix_sgf_sparse.x: smatrix_sgf_sparse.f90 kinds.o ai_overlap.o mathconstants.o orbtramat.o mathlib.o cgf_utils.o
	$(FC) $(FFLAGS) -o $@ $^ $(LIBS)

prescreen_smatrix_sgf_sparse.x: prescreen_smatrix_sgf_sparse.f90 kinds.o ai_overlap.o mathconstants.o orbtramat.o mathlib.o cgf_utils.o
	$(FC) $(FFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f *.o *.mod *.x
