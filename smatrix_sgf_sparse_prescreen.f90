PROGRAM smatrix_cgf_sparse
USE kinds, ONLY: dp, angstrom
USE cgf_utils, ONLY: atom_type, cgf_type, cgf_release, read_cgf_basis_set, &
                     norm_cgf_gto_interface, overlap_ab_cgf_interface, convert_matrix_cgf_to_sgf, add_shell
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

   integer, allocatable, dimension(:,:) :: bas
   real(kind=dp), allocatable, dimension(:) :: env, s_sparse
   integer, allocatable, dimension(:,:) :: list_ijd
   integer :: sij_offset, ij_idx, n_pairs, s_size, curr_env_offset, sij_size, max_sij_size, max_npgf_col, max_npgf_row
   real(kind=dp) :: r2
   real(kind=dp), parameter :: s_prescreen_thrs_squared = 10.0_dp**2 ! units ? should be bohr^2

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
   !
   ! todo : too much c, all 2d arrays are iterating on the slow variable -_-
   !
   max_npgf_col = 0
   max_npgf_row = 0
   ! unravel from OOP to libcint format
   allocate( bas(ncgf_col+ncgf_row,8) )
   ! todo this should be resizing arrays
   allocate( env(10000 * (ncgf_col+ncgf_row)) )
   curr_env_offset = 1
   do icgf_col = 1,ncgf_col
      call add_shell ( icgf_col, cgfs_col(icgf_col), env, curr_env_offset, bas, atoms_col(cgfs_col(icgf_col)%iatom) )
      ! todo: this should be a loop over the basis set, not over the atoms
      max_npgf_col = max( max_npgf_col, cgfs_col(icgf_col)%npgf )
   end do
   do icgf_row = 1,ncgf_row
      call add_shell ( icgf_row, cgfs_row(icgf_row), env, curr_env_offset, bas, atoms_row(cgfs_row(icgf_row)%iatom) )
      max_npgf_row = max( max_npgf_row, cgfs_row(icgf_row)%npgf )
   end do
   !
   ! prescreen
   ! for each pair, check if close enough. If yes, add i,j and where their
   ! smatrix is going to be stored
   ! todo this should be resizing arrays
   allocate( list_ijd(ncgf_col*ncgf_row,3) )
   sij_offset = 1 ! note: this is fortran, 0 is 1
   ij_idx = 1     !
   max_sij_size = 0
   DO icgf_col = 1,ncgf_col
      DO icgf_row = 1,ncgf_row
         rab = atoms_row(cgfs_row(icgf_row)%iatom)%r - atoms_col(cgfs_col(icgf_col)%iatom)%r
         r2 = dot_product(rab, rab)
         if ( r2 < s_prescreen_thrs_squared ) then 
            list_ijd(ij_idx,:) = (/ icgf_col, ncgf_col+icgf_row, sij_offset /)
            sij_size = get_nco(cgfs_col(icgf_col)%l) * get_nso(cgfs_row(icgf_row)%l)
            max_sij_size = max( max_sij_size, sij_size )
            sij_offset = sij_offset + sij_size
            ij_idx = ij_idx + 1
         end if
      end do
   end do
   n_pairs = ij_idx-1 ! confusing af. We want ij to be 1 when accessing array, and zero when counting the number of pairs
   s_size = sij_offset ! should this have also get a -1 ?
   allocate ( s_sparse(s_size) )

   call compute_s ( list_ijd, atm, bas, env, s_sparse )

!  
!  dim3 max_npgf_ab(max_npgf_col, mx_npgf_row)
!  compute_s_gpu<<< n_pairs, max_npgf_ab>>> ( int* list_ijd_dev, int* bas_dev, double* env_dev, double* s_sparse_dev )
!  {
!       //
!       int ijd_idx = blockIdx.x * PAL_SLOTS
!       int i = list_ijd_dev[ ijd_idx + 0 ] * BAS_SLOTS
!       int j = list_ijd_dev[ ijd_idx + 1 ] * BAS_SLOTS
!       int s_offset = list_ijd_dev[ ijd_idx + 2 ] // might be pushed to after the if, but it is more elegant here
!       int ipgf_a = threadIdx.x
!       int ipgf_b = threadIdx.y
!       int npgf_a = bas_dev[i+BAS_OFFSET_NPGF]
!       int npgf_b = bas_dev[j+BAS_OFFSET_NPGF]
!       // We size the block to accomodate the largest contractionso smaller contractions only use a subset of the threads
!       // so smaller contractions only use a subset of the threads
!       // worse case is a contraction with high angular moment and a single coefficient
!       // in which case one thread is doing all L calculations
!       if ( (ipgf_a<npgf_a) and(ipgf_b<npgf_b)) {
  !       int la = bas_dev[i+BAS_OFFSET_L]
  !       int lb = bas_dev[j+BAS_OFFSET_L]
  !       int ncoa = get_nco(la)
  !       int ncob = get_nco(lb)
  !       double* zet_a = &env_dev[ bas_dev[i+BAS_OFFSET_Z] ]
  !       double* zet_b = &env_dev[ bas_dev[j+BAS_OFFSET_Z] ]
  !       double* gcc_a = &env_dev[ bas_dev[i+BAS_OFFSET_C] ]
  !       double* gcc_b = &env_dev[ bas_dev[j+BAS_OFFSET_C] ]
  !       double ra_x = env_dev[ bas_dev[i+BAS_OFFSET_R] + 0 ]
  !       double ra_y = env_dev[ bas_dev[i+BAS_OFFSET_R] + 1 ]
  !       double ra_z = env_dev[ bas_dev[i+BAS_OFFSET_R] + 2 ]
  !       double rb_x = env_dev[ bas_dev[j+BAS_OFFSET_R] + 0 ]
  !       double rb_y = env_dev[ bas_dev[j+BAS_OFFSET_R] + 1 ]
  !       double rb_z = env_dev[ bas_dev[j+BAS_OFFSET_R] + 2 ]
  !       double rab_x = ra_x - rb_x
  !       double rab_y = ra_y - rb_y
  !       double rab_z = ra_z - rb_z
  !       double sab_pgf[ncoa*ncob] // if L = 6, this is ((6+1)*(6+2)/2)**2 = 784 doubles per thread. Not great
  !       double cSc_ab
  !       //
  !       // Compute the gaussian integrals and saves them in sab_pgf
  !       overlap_primitive_(la, lb, sab_pgf, zet_a[ipgf_a], zet_b[ipgf_b], rab_x, rab_y, rab_z)
  !       // Contract the gaussian integrals to the different products between basis set functions
  !       for (unsigned int icob = 0; icob < ncob; ++icob) {
  !         for (unsigned int icoa = 0; icoa < ncoa; ++icoa) {
  !           cSc_ab = sab_pgf_dev[icob*ncoa+icoa] *  gcc_a[icoa*npgf_a+ipgf_a] * gcc_b[icob*npgf_b+ipgf_b];
  !           // Thanks to s_offset, writes to sab_dev from different blocks will never overlap
  !           atomicAdd_block(&s_sparse_dev[s_offset + icob*ncoa+icoa], cSc_ab);
  !         }
  !       }
!       }
!    
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
