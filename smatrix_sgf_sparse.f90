PROGRAM smatrix_cgf_sparse
USE kinds, ONLY: dp, angstrom
USE cgf_utils, ONLY: atom_type, cgf_type, cgf_release, read_cgf_basis_set, &
                     norm_cgf_gto_interface, overlap_ab_cgf_interface, convert_matrix_cgf_to_sgf
USE orbtramat, ONLY: get_nco, get_nso
IMPLICIT NONE
   TYPE(atom_type), ALLOCATABLE, DIMENSION(:) :: atoms_row, atoms_col
   TYPE(cgf_type), ALLOCATABLE, DIMENSION(:)  :: cgfs_row, cgfs_col

   CHARACTER(len=512) :: cgf_in_row, cgf_in_col
   CHARACTER(len=16) :: ncols_str

   INTEGER :: icgf_col, ncgf_col, nco_col, nso_col
   INTEGER :: irow, icgf_row, ncgf_row, nco_row, nso_row

   REAL(kind=dp), PARAMETER :: threshold = 1.0e-10_dp
   REAL(kind=dp)                               :: norm
   REAL(kind=dp), DIMENSION(3)                 :: rab
   REAL(kind=dp), ALLOCATABLE, DIMENSION(:, :) :: sab_cgf, sab_sgf

   cgf_in_row = "01-NaCl-bulk_SZV-MOLOPT_row_unitcell.cgfs"
   cgf_in_col = "01-NaCl-bulk_SZV-MOLOPT_col_supercell.cgfs"

   CALL read_cgf_basis_set(TRIM(cgf_in_row), atoms_row, cgfs_row)
   DO irow = 1,SIZE(atoms_row)
      atoms_row(irow)%r(:) = atoms_row(irow)%r(:) / angstrom ! Angstroms -> atomic units (Bohr)
   END DO
   ncgf_row = SIZE(cgfs_row)
   DO icgf_row = 1,ncgf_row
      CALL norm_cgf_gto_interface(cgfs_row(icgf_row))
   END DO

   CALL read_cgf_basis_set(TRIM(cgf_in_col), atoms_col, cgfs_col)
   DO irow = 1,SIZE(atoms_col)
      atoms_col(irow)%r(:) = atoms_col(irow)%r(:) / angstrom ! Angstroms -> bohr
   END DO
   ncgf_col = SIZE(cgfs_col)
   DO icgf_col = 1,ncgf_col
      CALL norm_cgf_gto_interface(cgfs_col(icgf_col))
   END DO

   DO icgf_col = 1,ncgf_col
      nco_col = get_nco(cgfs_col(icgf_col)%l)
      nso_col = get_nso(cgfs_col(icgf_col)%l)

      DO icgf_row = 1,ncgf_row
         nco_row = get_nco(cgfs_row(icgf_row)%l)
         nso_row = get_nso(cgfs_row(icgf_row)%l)

         ALLOCATE (sab_cgf(nco_row, nco_col), sab_sgf(nso_row, nso_col))

         rab = atoms_row(cgfs_row(icgf_row)%iatom)%r - atoms_col(cgfs_col(icgf_col)%iatom)%r

         CALL overlap_ab_cgf_interface(cgfs_row(icgf_row), cgfs_col(icgf_col), rab, sab_cgf)

         ! Cartesian Gaussians -> Spherical Gaussians
         CALL convert_matrix_cgf_to_sgf(cgfs_row(icgf_row:icgf_row), cgfs_col(icgf_col:icgf_col), sab_cgf, sab_sgf)

         norm = MAXVAL(ABS(sab_sgf))
         IF (norm >= threshold) THEN
            WRITE(ncols_str,'(I0)') SIZE(sab_sgf, 2)

            WRITE(*,'(2I7)') icgf_row, icgf_col
            DO irow = 1,SIZE(sab_sgf, 1)
               WRITE(*,'('//TRIM(ncols_str)//'ES20.10E3)') sab_sgf(irow,:)
            END DO
         END IF

         DEALLOCATE (sab_cgf, sab_sgf)
      END DO
   END DO

   DO icgf_col = SIZE(cgfs_col),1,-1
      CALL cgf_release(cgfs_col(icgf_col))
   END DO
   DO icgf_row = SIZE(cgfs_row),1,-1
      CALL cgf_release(cgfs_row(icgf_row))
   END DO
   DEALLOCATE (cgfs_col, cgfs_row)
   DEALLOCATE (atoms_col, atoms_row)

END PROGRAM smatrix_cgf_sparse
