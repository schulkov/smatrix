MODULE cgf_utils
USE ISO_FORTRAN_ENV, ONLY: IOSTAT_END
USE ISO_C_BINDING, ONLY: C_DOUBLE, C_INT, C_LOC, C_PTR
USE kinds,  ONLY: dp
USE orbtramat, ONLY: get_nco, get_nso, get_c2s
IMPLICIT NONE
PRIVATE

   PUBLIC  :: atom_type, cgf_type, cgf_release

   PRIVATE :: read_line
   PUBLIC  :: read_cgf_basis_set

   PUBLIC  :: norm_cgf_gto_interface, overlap_ab_cgf_interface
   PUBLIC  :: convert_matrix_cgf_to_sgf

   TYPE atom_type
      ! atomic nuclear charge
      INTEGER                                    :: atomic_number
      ! atomic coordinates in atomic units (bohrs)
      REAL(kind=dp), DIMENSION(3)                :: r
   END TYPE atom_type

   TYPE cgf_type
      ! atomic index (1 .. number of atoms), angular momentum, number of primitive GTOs
      INTEGER                                    :: iatom, l, npgf
      ! exponent, Gaussian contraction coefficients (normalisation factor should already been applied to the primitive GTO),
      REAL(kind=dp), ALLOCATABLE, DIMENSION(:)   :: zet, gcc
      ! normalisation coefficient applied to unnormalised Gaussian that normalises the contracted GTO
      REAL(kind=dp), ALLOCATABLE, DIMENSION(:,:) :: gcc_total
   END TYPE cgf_type

   ! explicit interface for external C functions
   INTERFACE
      SUBROUTINE overlap_ab_cgf(sab, la_set, npgf_a, zet_a, gcc_a, lb_set, npgf_b, zet_b, &
                                gcc_b, rab_x, rab_y, rab_z) BIND(C, name="overlap_ab_cgf")
         IMPORT :: C_DOUBLE, C_INT, C_PTR
         TYPE(C_PTR), VALUE :: sab
         INTEGER(kind=C_INT), VALUE :: la_set, npgf_a, lb_set, npgf_b
         REAL(kind=C_DOUBLE), VALUE :: rab_x, rab_y, rab_z
         TYPE(C_PTR), VALUE :: zet_a, zet_b, gcc_a, gcc_b
      END SUBROUTINE overlap_ab_cgf

      SUBROUTINE norm_cgf_gto(l_set, npgf, zet, gcc, gcc_total) BIND(C, name="norm_cgf_gto")
         IMPORT :: C_INT, C_PTR
         INTEGER(kind=C_INT), VALUE :: l_set, npgf
         TYPE(C_PTR), VALUE :: zet, gcc, gcc_total
      END SUBROUTINE norm_cgf_gto
   END INTERFACE

CONTAINS

   SUBROUTINE cgf_release(cgf)
      TYPE(cgf_type), INTENT(inout)              :: cgf

      cgf%npgf = 0

      IF (ALLOCATED(cgf%zet)) DEALLOCATE (cgf%zet)
      IF (ALLOCATED(cgf%gcc)) DEALLOCATE (cgf%gcc)
      IF (ALLOCATED(cgf%gcc_total)) DEALLOCATE (cgf%gcc_total)
   END SUBROUTINE cgf_release

   SUBROUTINE read_line(funit, line, eof)
      INTEGER, INTENT(in)                                       :: funit
      CHARACTER(len=*), INTENT(out)                             :: line
      LOGICAL, INTENT(out), OPTIONAL                            :: eof

      INTEGER :: pos, stat

      IF (PRESENT(eof)) eof = .FALSE.

      DO
         READ (funit, '(A)', iostat=stat) line
         IF (stat == IOSTAT_END) THEN
            IF (PRESENT(eof)) THEN
               eof = .TRUE.
               EXIT
            ELSE
               ! I/O error; the next line throws a Fortran runtime exception to terminate the program
               READ (funit, '(A)') line
            END IF
         END IF

         ! remove comments starting with '#'
         pos = SCAN (line, '#')
         IF (pos > 0) WRITE (line(pos:), *)

         ! skip empty line
         IF (LEN_TRIM(line) > 0) EXIT
      END DO

   END SUBROUTINE read_line

   SUBROUTINE read_cgf_basis_set(filename, atoms, cgfs)
      CHARACTER(len=*), INTENT(in)                              :: filename
      TYPE(atom_type), ALLOCATABLE, DIMENSION(:), INTENT(inout) :: atoms
      TYPE(cgf_type), ALLOCATABLE, DIMENSION(:), INTENT(inout)  :: cgfs

      INTEGER, PARAMETER :: funit = 100

      CHARACTER(len=256)                                        :: line
      INTEGER :: iatom, icgf, ipgf, l, natoms, ncgf, npgf

      OPEN (funit, file=TRIM(ADJUSTL(filename)), status="old", action="read")

      ! Atomic coordinates (in Bohr)
      !  * number of atoms
      CALL read_line(funit, line)
      READ (line, *) natoms

      !  * atomic coordinates
      ALLOCATE (atoms(natoms))
      DO iatom = 1,natoms
         CALL read_line(funit, line)
         READ (line, *) atoms(iatom)%atomic_number, atoms(iatom)%r(:)
      END DO

      ! Cartesian GTO-s
      !  * number of cartesian gaussian functions
      CALL read_line(funit, line)
      READ (line, *) ncgf

      !  * cgfs
      ALLOCATE (cgfs(ncgf))
      DO icgf = 1,ncgf
         ! ** atomic index, wfx type, number of primitive gaussian functions
         CALL read_line(funit, line)
         READ (line, *) iatom, l, npgf
         cgfs(icgf)%iatom = iatom
         cgfs(icgf)%l = l
         cgfs(icgf)%npgf = npgf
         ALLOCATE (cgfs(icgf)%zet(npgf), cgfs(icgf)%gcc(npgf), cgfs(icgf)%gcc_total(npgf, get_nco(l)))

         ! ** zet, unnormalised contraction coefficient
         DO ipgf = 1,npgf
            CALL read_line(funit, line)
            READ (line, *) cgfs(icgf)%zet(ipgf), cgfs(icgf)%gcc(ipgf)
         END DO

         cgfs(icgf)%gcc_total = 0.0_dp
      END DO

      CLOSE (funit)
   END SUBROUTINE read_cgf_basis_set

   SUBROUTINE norm_cgf_gto_interface(cgf)
      TYPE(cgf_type), INTENT(inout), TARGET      :: cgf

      CALL norm_cgf_gto(cgf%l, cgf%npgf, C_LOC(cgf%zet(1)), C_LOC(cgf%gcc(1)), C_LOC(cgf%gcc_total(1,1)))
   END SUBROUTINE norm_cgf_gto_interface


   SUBROUTINE overlap_ab_cgf_interface(cgfa, cgfb, rab, sab)
      TYPE(cgf_type), INTENT(in), TARGET         :: cgfa, cgfb
      REAL(kind=dp), DIMENSION(3), INTENT(in)    :: rab
      REAL(kind=dp), DIMENSION(:,:), INTENT(inout), TARGET :: sab

      CALL overlap_ab_cgf(C_LOC(sab(1,1)), cgfa%l, cgfa%npgf, C_LOC(cgfa%zet(1)), C_LOC(cgfa%gcc_total(1,1)), &
                          cgfb%l, cgfb%npgf, C_LOC(cgfb%zet(1)), C_LOC(cgfb%gcc_total(1,1)), &
                          rab(1), rab(2), rab(3))

   END SUBROUTINE overlap_ab_cgf_interface

   SUBROUTINE convert_matrix_cgf_to_sgf(cgfs_row, cgfs_col, sab_cgf, sab_sgf)
      TYPE(cgf_type), DIMENSION(:), INTENT(in)   :: cgfs_row, cgfs_col
      REAL(kind=dp), DIMENSION(:,:), INTENT(in)  :: sab_cgf
      REAL(kind=dp), DIMENSION(:,:), INTENT(out) :: sab_sgf

      TYPE c2s_matrix_type
         REAL(kind=dp), ALLOCATABLE, DIMENSION(:,:) :: c2s
      END TYPE c2s_matrix_type
      TYPE(c2s_matrix_type), ALLOCATABLE, DIMENSION(:) :: c2s_matrices

      INTEGER :: icgf_col, ncgf_col, nco_col, nco_col_total, nso_col, nso_col_total
      INTEGER :: icgf_row, ncgf_row, nco_row, nco_row_total, nso_row, nso_row_total
      INTEGER :: l, lmax

      lmax = 0

      ncgf_row = SIZE(cgfs_row)
      nco_row = 0
      nso_row = 0
      DO icgf_row = 1,ncgf_row
         l = cgfs_row(icgf_row)%l
         IF (lmax < l) lmax = l
         nco_row = nco_row + get_nco(l)
         nso_row = nso_row + get_nso(l)
      END DO

      ncgf_col = SIZE(cgfs_col)
      nco_col = 0
      nso_col = 0
      DO icgf_col = 1,ncgf_col
         l = cgfs_col(icgf_col)%l
         IF (lmax < l) lmax = l
         nco_col = nco_col + get_nco(l)
         nso_col = nso_col + get_nso(l)
      END DO

      ALLOCATE (c2s_matrices(0:lmax))
      DO l = 0,lmax
         ALLOCATE(c2s_matrices(l)%c2s(get_nso(l), get_nco(l)))
         c2s_matrices(l)%c2s = get_c2s(l)
      END DO

      nco_col_total = 0
      nso_col_total = 0
      DO icgf_col = 1,ncgf_col
         nco_col = get_nco(cgfs_col(icgf_col)%l)
         nso_col = get_nso(cgfs_col(icgf_col)%l)

         nco_row_total = 0
         nso_row_total = 0
         DO icgf_row = 1,ncgf_row
            nco_row = get_nco(cgfs_row(icgf_row)%l)
            nso_row = get_nso(cgfs_row(icgf_row)%l)

            sab_sgf(nso_row_total+1:nso_row_total+nso_row, nso_col_total+1:nso_col_total+nso_col) = &
              MATMUL(c2s_matrices(cgfs_row(icgf_row)%l)%c2s, &
                 MATMUL(sab_cgf(nco_row_total+1:nco_row_total+nco_row, nco_col_total+1:nco_col_total+nco_col), &
                    TRANSPOSE(c2s_matrices(cgfs_col(icgf_col)%l)%c2s)))

            nco_row_total = nco_row_total + nco_row
            nso_row_total = nso_row_total + nso_row
         END DO

         nco_col_total = nco_col_total + nco_col
         nso_col_total = nso_col_total + nso_col
      END DO

      DO l = lmax,0,-1
         DEALLOCATE (c2s_matrices(l)%c2s)
      END DO
      DEALLOCATE (c2s_matrices)
   END SUBROUTINE convert_matrix_cgf_to_sgf
END MODULE cgf_utils
