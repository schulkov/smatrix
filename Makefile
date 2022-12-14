FC = gfortran
FFLAGS = -O3 -Wall -std=f2008
CC = gcc
CFLAGS = -O3 -Wall

# Avoid funny character set dependencies
unexport LC_ALL
LC_COLLATE=C
LC_NUMERIC=C
export LC_COLLATE LC_NUMERIC

# Avoid interference with shell env settings
unexport GREP_OPTIONS

PHONY := make clean
make: smatrix_sgf_dense.x smatrix_sgf_sparse.x

kinds.o: kinds.f90
	$(FC) $(FFLAGS) -c -o $@ $<

mathconstants.o: mathconstants.f90 kinds.o
	$(FC) $(FFLAGS) -c -o $@ $<

mathlib.o: mathlib.f90 mathconstants.o
	$(FC) $(FFLAGS) -c -o $@ $<

orbtramat.o: orbtramat.f90 mathconstants.o mathlib.o
	$(FC) $(FFLAGS) -c -o $@ $<

ai_overlap.o: ai_overlap.c mathconstants.o orbtramat.o
	$(CC) $(CFLAGS) -c -o $@ $<

cgf_utils.o: cgf_utils.f90 kinds.o orbtramat.o ai_overlap.o
	$(FC) $(FFLAGS) -c -o $@ $<

smatrix_sgf_dense.x: smatrix_sgf_dense.f90 kinds.o ai_overlap.o mathconstants.o orbtramat.o mathlib.o cgf_utils.o
	$(FC) $(FFLAGS) -o $@ $^

smatrix_sgf_sparse.x: smatrix_sgf_sparse.f90 kinds.o ai_overlap.o mathconstants.o orbtramat.o mathlib.o cgf_utils.o
	$(FC) $(FFLAGS) -o $@ $^

clean:
	rm -f *.o *.mod *.x
