PROGRAM smatrix_cgf_sparse
USE kinds, ONLY: dp, angstrom
USE cgf_utils, ONLY: atom_type, cgf_type, cgf_release, read_cgf_basis_set, compute_s_gpu, &
                     norm_cgf_gto_interface, overlap_ab_cgf_interface, convert_matrix_cgf_to_sgf, add_shell
USE orbtramat, ONLY: get_nco, get_nso
IMPLICIT NONE
   TYPE(atom_type), ALLOCATABLE, DIMENSION(:) :: atoms_row, atoms_col
   TYPE(cgf_type), ALLOCATABLE, DIMENSION(:)  :: cgfs_row, cgfs_col

   CHARACTER(len=512) :: cgf_in_row, cgf_in_col


   INTEGER :: icgf_col, ncgf_col, nco_col, nso_col
   INTEGER :: irow, icgf_row, ncgf_row, nco_row, nso_row



   REAL(kind=dp), DIMENSION(3)                 :: rab
   REAL(kind=dp), ALLOCATABLE, DIMENSION(:, :) :: sab_cgf, sab_sgf

   integer, allocatable, dimension(:,:) :: bas
   real(kind=dp), allocatable, dimension(:) :: env, s_sparse
   integer, allocatable, dimension(:,:) :: list_ijd
   integer :: sij_offset, ij_idx, n_pairs, s_size, curr_env_offset, sij_size, max_sij_size, max_npgf_col, max_npgf_row
   real(kind=dp) :: r2
   real(kind=dp), parameter :: s_prescreen_thrs_squared = 30.0_dp**2 ! units ? should be bohr^2
   integer :: co_col, co_row, pair_idx, idx
   real(kind=dp) :: val ,ref
   ! read input files
!   cgf_in_row = "01-NaCl-bulk_SZV-MOLOPT_col_supercell.cgfs"
!   cgf_in_col = "01-NaCl-bulk_SZV-MOLOPT_col_supercell.cgfs"
   cgf_in_row = "02-NaCl-bulk_SZV-MOLOPT_col_supercell.cgfs"
   cgf_in_col = "02-NaCl-bulk_SZV-MOLOPT_col_supercell.cgfs"



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
   ! unravel from OOP to libcint format
   max_npgf_col = 0
   max_npgf_row = 0
   allocate( bas(8,ncgf_col+ncgf_row) )
   ! todo this should be resizing arrays
   allocate( env(10000 * (ncgf_col+ncgf_row)) )
   curr_env_offset = 1
   do icgf_col = 1,ncgf_col
      call add_shell ( icgf_col, cgfs_col(icgf_col), env, curr_env_offset, bas, atoms_col(cgfs_col(icgf_col)%iatom) )
      ! todo: this should be a loop over the basis set, not over the atoms
      max_npgf_col = max( max_npgf_col, cgfs_col(icgf_col)%npgf )
   end do
   do icgf_row = 1,ncgf_row
      call add_shell ( icgf_row+ncgf_col, cgfs_row(icgf_row), env, curr_env_offset, bas, atoms_row(cgfs_row(icgf_row)%iatom) )
      max_npgf_row = max( max_npgf_row, cgfs_row(icgf_row)%npgf )
   end do
   !
   ! prescreen
   ! for each pair, check if close enough. If yes, add i,j and where their smatrix is going to be stored
   ! todo this should be resizing arrays
   allocate( list_ijd(3,ncgf_col*ncgf_row) )
   sij_offset = 1
   ij_idx = 1
   max_sij_size = 0
   DO icgf_col = 1,ncgf_col
      DO icgf_row = 1,ncgf_row
         rab = atoms_row(cgfs_row(icgf_row)%iatom)%r - atoms_col(cgfs_col(icgf_col)%iatom)%r
         r2 = dot_product(rab, rab)
         if ( r2 < s_prescreen_thrs_squared ) then 
            list_ijd(:,ij_idx) = (/ icgf_col, ncgf_col+icgf_row, sij_offset /)
            sij_size = get_nco(cgfs_col(icgf_col)%l) * get_nco(cgfs_row(icgf_row)%l)
            max_sij_size = max( max_sij_size, sij_size )
            sij_offset = sij_offset + sij_size
            ij_idx = ij_idx + 1
         end if
      end do
   end do
   n_pairs = ij_idx-1  ! confusing af. We want ij to be 1 when accessing array, and zero when counting the number of pairs
   s_size = sij_offset ! should this have also get a -1 ?
   allocate ( s_sparse(s_size) )
   print *, 'computing s with ', n_pairs, ' pairs out of ', ncgf_col*ncgf_row, ' possible pairs'
   print *, ' s size: ', s_size
   ! work
   call compute_s_gpu ( list_ijd, bas, env, s_sparse, &
                        n_pairs, ncgf_col+ncgf_row, curr_env_offset, s_size, max_npgf_col, max_npgf_row )
   ! check
   do pair_idx=1, n_pairs
      icgf_col = list_ijd(1, pair_idx)
      icgf_row = list_ijd(2, pair_idx) - ncgf_col
      sij_offset = list_ijd(3, pair_idx)
      nco_col = get_nco(cgfs_col(icgf_col)%l)
      nco_row = get_nco(cgfs_row(icgf_row)%l)
      nso_col = get_nso(cgfs_col(icgf_col)%l)
      nso_row = get_nso(cgfs_row(icgf_row)%l)
      ALLOCATE (sab_cgf(nco_row, nco_col), sab_sgf(nso_row,nso_col))
      rab = atoms_row(cgfs_row(icgf_row)%iatom)%r - atoms_col(cgfs_col(icgf_col)%iatom)%r
      CALL overlap_ab_cgf_interface(cgfs_row(icgf_row), cgfs_col(icgf_col), rab, sab_cgf)
      ! Cartesian Gaussians -> Spherical Gaussians
      CALL convert_matrix_cgf_to_sgf(cgfs_row(icgf_row:icgf_row), cgfs_col(icgf_col:icgf_col), sab_cgf, sab_sgf)
      do co_row=1, nco_row
         do co_col=1, nco_col
            idx = sij_offset+(co_row-1)*nco_col+co_col-1
            val = s_sparse(idx)
            ref = sab_cgf(co_row, co_col)
            if ( (val-ref)**2 > 1.e-12 ) then
              print *, 'wrong value at icgf_col, icgf_row, li, lj, co_row, co_col, idx, val , ref: ', & 
                                       icgf_col, icgf_row, cgfs_col(icgf_col)%l , cgfs_row(icgf_row)%l, &
                                                                   co_row, co_col, idx, val , ref
              end if
         end do
      end do
      deallocate(sab_cgf)
      deallocate(sab_sgf)
   end do

   DO icgf_col = SIZE(cgfs_col),1,-1
      CALL cgf_release(cgfs_col(icgf_col))
   END DO
   DO icgf_row = SIZE(cgfs_row),1,-1
      CALL cgf_release(cgfs_row(icgf_row))
   END DO
   DEALLOCATE (cgfs_col, cgfs_row)
   DEALLOCATE (atoms_col, atoms_row)

END PROGRAM smatrix_cgf_sparse
