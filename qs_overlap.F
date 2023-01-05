!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2022 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Calculation of overlap matrix, its derivatives and forces
!> \par History
!>      JGH: removed printing routines
!>      JGH: upgraded to unique routine for overlaps
!>      JGH: Add specific routine for 'forces only'
!>           Major refactoring for new overlap routines
!>      JGH: Kpoints
!>      JGH: openMP refactoring
!> \author Matthias Krack (03.09.2001,25.06.2003)
! **************************************************************************************************
MODULE qs_overlap
   USE ai_contraction,                  ONLY: block_add,&
                                              contraction,&
                                              decontraction,&
                                              force_trace
   USE ai_overlap,                      ONLY: overlap_ab
   USE atomic_kind_types,               ONLY: atomic_kind_type,&
                                              get_atomic_kind_set
   USE basis_set_types,                 ONLY: get_gto_basis_set,&
                                              gto_basis_set_p_type,&
                                              gto_basis_set_type
   USE block_p_types,                   ONLY: block_p_type
   USE cp_control_types,                ONLY: dft_control_type
   USE cp_dbcsr_cp2k_link,              ONLY: cp_dbcsr_alloc_block_from_nbl
   USE cp_dbcsr_operations,             ONLY: dbcsr_allocate_matrix_set
   USE dbcsr_api,                       ONLY: &
        dbcsr_create, dbcsr_distribution_type, dbcsr_filter, dbcsr_finalize, dbcsr_get_block_p, &
        dbcsr_p_type, dbcsr_type, dbcsr_type_antisymmetric, dbcsr_type_no_symmetry, &
        dbcsr_type_symmetric
   USE kinds,                           ONLY: default_string_length,&
                                              dp,&
                                              int_8
   USE kpoint_types,                    ONLY: get_kpoint_info,&
                                              kpoint_type
   USE orbital_pointers,                ONLY: indco,&
                                              ncoset
   USE orbital_symbols,                 ONLY: cgf_symbol
   USE particle_methods,                ONLY: get_particle_set
   USE particle_types,                  ONLY: particle_type
   USE qs_force_types,                  ONLY: qs_force_type
   USE qs_integral_utils,               ONLY: basis_set_list_setup,&
                                              get_memory_usage
   USE qs_kind_types,                   ONLY: qs_kind_type
   USE qs_ks_types,                     ONLY: get_ks_env,&
                                              qs_ks_env_type
   USE qs_neighbor_list_types,          ONLY: get_neighbor_list_set_p,&
                                              neighbor_list_set_p_type
   USE string_utilities,                ONLY: compress,&
                                              uppercase
   USE virial_methods,                  ONLY: virial_pair_force
   USE virial_types,                    ONLY: virial_type
   USE iso_c_binding, only: c_int, c_ptr, c_loc
   
!$ USE OMP_LIB, ONLY: omp_lock_kind, &
!$                    omp_init_lock, omp_set_lock, &
!$                    omp_unset_lock, omp_destroy_lock

#include "./base/base_uses.f90"

   IMPLICIT NONE
   INTERFACE
      subroutine overlap_ab_gpu( la_max, la_min, npgfa, rpgfa, zeta, &
            lb_max, lb_min, npgfb, rpgfb, zetb, rab, sab, lds ) &
            bind(C,name="overlap_ab_gpu")
         import :: c_int, c_ptr
         integer(kind=c_int), value :: la_max, la_min, npgfa, lb_max, lb_min, npgfb, lds
         type(c_ptr), value :: rpgfa, zeta, rpgfb, zetb, rab, sab
      end subroutine overlap_ab_gpu
   END INTERFACE

   PRIVATE

! *** Global parameters ***

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'qs_overlap'

   ! should be a parameter, but this triggers a bug in OMPed code with gfortran 4.9
   INTEGER, DIMENSION(1:56), SAVE :: ndod = (/0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, &
                                              1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, &
                                              0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, &
                                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1/)

   INTERFACE create_sab_matrix
      MODULE PROCEDURE create_sab_matrix_1d, create_sab_matrix_2d
   END INTERFACE

! *** Public subroutines ***

   PUBLIC :: build_overlap_matrix, build_overlap_matrix_simple, &
             build_overlap_force, create_sab_matrix

CONTAINS

! **************************************************************************************************
!> \brief   Calculation of the overlap matrix over Cartesian Gaussian functions.
!> \param   ks_env the QS env
!> \param   matrix_s The overlap matrix to be calculated (optional)
!> \param   matrixkp_s The overlap matrices to be calculated (kpoints, optional)
!> \param   matrix_name The name of the overlap matrix (i.e. for output)
!> \param   nderivative Derivative with respect to basis origin
!> \param   basis_type_a basis set to be used for bra in <a|b>
!> \param   basis_type_b basis set to be used for ket in <a|b>
!> \param   sab_nl pair list (must be consistent with basis sets!)
!> \param   calculate_forces (optional) ...
!> \param   matrix_p density matrix for force calculation (optional)
!> \param   matrixkp_p density matrix for force calculation with k_points (optional)
!> \date    11.03.2002
!> \par     History
!>          Enlarged functionality of this routine. Now overlap matrices based
!>          on different basis sets can be calculated, taking into account also
!>          mixed overlaps NOTE: the pointer to the overlap matrix must now be
!>          put into its corresponding env outside of this routine
!>          [Manuel Guidon]
!>          Generalized for derivatives and force calculations [JHU]
!>          Kpoints, returns overlap matrices in real space index form
!> \author  MK
!> \version 1.0
! **************************************************************************************************
   SUBROUTINE build_overlap_matrix(ks_env, matrix_s, matrixkp_s, matrix_name, &
                                   nderivative, basis_type_a, basis_type_b, sab_nl, calculate_forces, &
                                   matrix_p, matrixkp_p)

      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      TYPE(dbcsr_p_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: matrix_s
      TYPE(dbcsr_p_type), DIMENSION(:, :), OPTIONAL, &
         POINTER                                         :: matrixkp_s
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL             :: matrix_name
      INTEGER, INTENT(IN), OPTIONAL                      :: nderivative
      CHARACTER(LEN=*), INTENT(IN)                       :: basis_type_a, basis_type_b
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: sab_nl
      LOGICAL, INTENT(IN), OPTIONAL                      :: calculate_forces
      TYPE(dbcsr_type), OPTIONAL, POINTER                :: matrix_p
      TYPE(dbcsr_p_type), DIMENSION(:, :), OPTIONAL, &
         POINTER                                         :: matrixkp_p

      INTEGER                                            :: natom

      MARK_USED(int_8)

      CALL get_ks_env(ks_env, natom=natom)

      CALL build_overlap_matrix_low(ks_env, matrix_s, matrixkp_s, matrix_name, nderivative, &
                                    basis_type_a, basis_type_b, sab_nl, calculate_forces, &
                                    matrix_p, matrixkp_p, natom)

   END SUBROUTINE build_overlap_matrix

! **************************************************************************************************
!> \brief Implementation of build_overlap_matrix with the additional natom argument passed in to
!>        allow for explicitly shaped force_thread array which is needed for OMP REDUCTION.
!> \param ks_env ...
!> \param matrix_s ...
!> \param matrixkp_s ...
!> \param matrix_name ...
!> \param nderivative ...
!> \param basis_type_a ...
!> \param basis_type_b ...
!> \param sab_nl ...
!> \param calculate_forces ...
!> \param matrix_p ...
!> \param matrixkp_p ...
!> \param natom ...
! **************************************************************************************************
   SUBROUTINE build_overlap_matrix_low(ks_env, matrix_s, matrixkp_s, matrix_name, nderivative, &
                                       basis_type_a, basis_type_b, sab_nl, calculate_forces, &
                                       matrix_p, matrixkp_p, natom)

      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      TYPE(dbcsr_p_type), DIMENSION(:), OPTIONAL, &
         POINTER                                         :: matrix_s
      TYPE(dbcsr_p_type), DIMENSION(:, :), OPTIONAL, &
         POINTER                                         :: matrixkp_s
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL             :: matrix_name
      INTEGER, INTENT(IN), OPTIONAL                      :: nderivative
      CHARACTER(LEN=*), INTENT(IN)                       :: basis_type_a, basis_type_b
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: sab_nl
      LOGICAL, INTENT(IN), OPTIONAL                      :: calculate_forces
      TYPE(dbcsr_type), OPTIONAL, POINTER                :: matrix_p
      TYPE(dbcsr_p_type), DIMENSION(:, :), OPTIONAL, &
         POINTER                                         :: matrixkp_p
      INTEGER, INTENT(IN)                                :: natom

      CHARACTER(len=*), PARAMETER :: routineN = 'build_overlap_matrix_low'

      INTEGER :: atom_a, handle, i, iatom, ic, icol, ikind, irow, iset, jatom, jkind, jset, ldsab, &
         maxder, maxs, n1, n2, ncoa, ncob, nder, nimg, nkind, nseta, nsetb, sgfa, sgfb, slot
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: atom_of_kind, kind_of
      INTEGER, DIMENSION(3)                              :: cell
      INTEGER, DIMENSION(:), POINTER                     :: la_max, la_min, lb_max, lb_min, npgfa, &
                                                            npgfb, nsgfa, nsgfb
      INTEGER, DIMENSION(:, :), POINTER                  :: first_sgfa, first_sgfb
      INTEGER, DIMENSION(:, :, :), POINTER               :: cell_to_index
      LOGICAL                                            :: do_forces, do_symmetric, dokp, found, &
                                                            trans, use_cell_mapping, use_virial
      REAL(KIND=dp)                                      :: dab, f, f0, ff, rab2
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: owork, pmat
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :)     :: oint
      REAL(KIND=dp), DIMENSION(3), target                        :: force_a, rab
      REAL(KIND=dp), DIMENSION(3, 3)                     :: pv_thread
      REAL(KIND=dp), DIMENSION(3, natom)                 :: force_thread
      REAL(KIND=dp), DIMENSION(:), POINTER               :: set_radius_a, set_radius_b
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: p_block, rpgfa, rpgfb, scon_a, scon_b, &
                                                            zeta, zetb
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(block_p_type), ALLOCATABLE, DIMENSION(:)      :: sint
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(gto_basis_set_p_type), DIMENSION(:), POINTER  :: basis_set_list_a, basis_set_list_b
      TYPE(gto_basis_set_type), POINTER                  :: basis_set_a, basis_set_b
      TYPE(kpoint_type), POINTER                         :: kpoints
      TYPE(qs_force_type), DIMENSION(:), POINTER         :: force
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(virial_type), POINTER                         :: virial

      integer                                            :: offset, offset_to_add
      real(kind=dp), allocatable, dimension(:)           :: env
      integer, allocatable, dimension(:,:)               :: bas_a, atm_a, bas_b, atm_b

      integer :: nsetab, sab_dest, qab_dest, set_slot, ma, mb, offset_ai, offset_bj, r_offset
      integer :: la_max_iset, la_min_iset, npgfa_iset, nsgfa_iset, z_offseta, c_offseta
      integer :: lb_max_jset, lb_min_jset, npgfb_jset, nsgfb_jset, z_offsetb, c_offsetb
      real(kind=dp) , allocatable, dimension(:), target :: zeta_iset, rpgfa_iset, scon_a_iset
      real(kind=dp) , allocatable, dimension(:), target :: zetb_jset, rpgfb_jset, scon_b_jset
      
      integer, allocatable, dimension (:,:) :: sab_nl_set
      real(kind=dp) , allocatable, dimension(:,:), target :: temp_sab_matrix, temp_qab_matrix, ca, cb, temp_sab_matrix_cpu 

      real(kind=dp) , allocatable, dimension(:) :: temp_sab, temp_qab
      integer :: iset_iset, jset_jset
      real(kind=dp) , allocatable, dimension(:,:) :: temp_qab_matrix_ref
      real(kind=dp), allocatable, dimension(:)           :: old_env

!$    INTEGER(kind=omp_lock_kind), &
!$       ALLOCATABLE, DIMENSION(:) :: locks
!$    INTEGER                                            :: lock_num, hash, hash1, hash2
!$    INTEGER(KIND=int_8)                                :: iatom8
!$    INTEGER, PARAMETER                                 :: nlock = 501

      !call build_overlap_matrix_simple(ks_env, matrix_s,basis_set_list_a, basis_set_list_b, sab_nl)

      CALL timeset(routineN, handle)

      ! test for matrices (kpoints or standard gamma point)
      IF (PRESENT(matrix_s)) THEN
         dokp = .FALSE.
         use_cell_mapping = .FALSE.
      ELSEIF (PRESENT(matrixkp_s)) THEN
         dokp = .TRUE.
         CALL get_ks_env(ks_env=ks_env, kpoints=kpoints)
         CALL get_kpoint_info(kpoint=kpoints, cell_to_index=cell_to_index)
         use_cell_mapping = (SIZE(cell_to_index) > 1)
      ELSE
         CPABORT("")
      END IF

      NULLIFY (atomic_kind_set)
      CALL get_ks_env(ks_env, &
                      atomic_kind_set=atomic_kind_set, &
                      qs_kind_set=qs_kind_set, &
                      dft_control=dft_control)

      nimg = dft_control%nimages
      nkind = SIZE(qs_kind_set)

      IF (PRESENT(calculate_forces)) THEN
         do_forces = calculate_forces
      ELSE
         do_forces = .FALSE.
      END IF

      IF (PRESENT(nderivative)) THEN
         nder = nderivative
      ELSE
         nder = 0
      END IF
      maxder = ncoset(nder)

      ! check for symmetry
      CPASSERT(SIZE(sab_nl) > 0)
      CALL get_neighbor_list_set_p(neighbor_list_sets=sab_nl, symmetric=do_symmetric)
      IF (do_symmetric) THEN
         CPASSERT(basis_type_a == basis_type_b)
      END IF

      ! set up basis set lists
      ALLOCATE (basis_set_list_a(nkind), basis_set_list_b(nkind))
      CALL basis_set_list_setup(basis_set_list_a, basis_type_a, qs_kind_set)
      CALL basis_set_list_setup(basis_set_list_b, basis_type_b, qs_kind_set)

      IF (dokp) THEN
         CALL dbcsr_allocate_matrix_set(matrixkp_s, maxder, nimg)
         CALL create_sab_matrix(ks_env, matrixkp_s, matrix_name, basis_set_list_a, basis_set_list_b, &
                                sab_nl, do_symmetric)
      ELSE
         CALL dbcsr_allocate_matrix_set(matrix_s, maxder)
         CALL create_sab_matrix(ks_env, matrix_s, matrix_name, basis_set_list_a, basis_set_list_b, &
                                sab_nl, do_symmetric)
      END IF
      maxs = maxder

      use_virial = .FALSE.
      IF (do_forces) THEN
         CALL get_ks_env(ks_env=ks_env, force=force, virial=virial)
         use_virial = virial%pv_availability .AND. (.NOT. virial%pv_numer)
      END IF

      force_thread = 0.0_dp
      pv_thread = 0.0_dp

      ldsab = get_memory_usage(qs_kind_set, basis_type_a, basis_type_b)
      IF (do_forces) THEN
         ! we need density matrix for forces
         IF (dokp) THEN
            CPASSERT(PRESENT(matrixkp_p))
         ELSE
            CPASSERT(PRESENT(matrix_p))
         END IF
         nder = MAX(nder, 1)
      END IF
      maxder = ncoset(nder)


!$OMP PARALLEL DEFAULT(NONE) &
!$OMP SHARED (do_forces, ldsab, maxder, use_cell_mapping, do_symmetric, maxs, dokp, &
!$OMP         ncoset, nder, use_virial, ndod, sab_nl, &
!$OMP         matrix_s, matrix_p,basis_set_list_a, basis_set_list_b, cell_to_index, &
!$OMP         matrixkp_s, matrixkp_p, locks, natom, nkind)  &
!$OMP PRIVATE (oint, owork, pmat, sint, ikind, jkind, iatom, jatom, rab, cell, &
!$OMP          basis_set_a, basis_set_b, &
!$OMP          first_sgfa, la_max, la_min, npgfa, nsgfa, nseta, rpgfa, set_radius_a, ncoa, ncob, force_a, &
!$OMP          zeta, first_sgfb, lb_max, lb_min, npgfb, nsetb, rpgfb, set_radius_b, nsgfb, p_block, dab, f,  &
!$OMP          zetb, scon_a, scon_b, ic, irow, icol, f0, ff, found, trans, rab2, n1, n2, sgfa, sgfb, iset, jset, &
!$OMP          hash, hash1, hash2, iatom8, slot, ma, mb, temp_qab_matrix_ref ) &
!$OMP REDUCTION (+ : pv_thread, force_thread )

!$OMP SINGLE
!$    ALLOCATE (locks(nlock))
!$OMP END SINGLE

!$OMP DO
!$    DO lock_num = 1, nlock
!$       call omp_init_lock(locks(lock_num))
!$    END DO
!$OMP END DO

      NULLIFY (p_block)
      ALLOCATE (oint(ldsab, ldsab, maxder), owork(ldsab, ldsab))
      IF (do_forces) ALLOCATE (pmat(ldsab, ldsab))
      ALLOCATE (sint(maxs))
      DO i = 1, maxs
         NULLIFY (sint(i)%block)
      END DO

!$OMP DO SCHEDULE(GUIDED)

      DO slot = 1, sab_nl(1)%nl_size

         ikind = sab_nl(1)%nlist_task(slot)%ikind
         jkind = sab_nl(1)%nlist_task(slot)%jkind
         iatom = sab_nl(1)%nlist_task(slot)%iatom
         jatom = sab_nl(1)%nlist_task(slot)%jatom
         cell(:) = sab_nl(1)%nlist_task(slot)%cell(:)
         rab(1:3) = sab_nl(1)%nlist_task(slot)%r(1:3)

         basis_set_a => basis_set_list_a(ikind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_a)) CYCLE
         basis_set_b => basis_set_list_b(jkind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_b)) CYCLE

!$       iatom8 = INT(iatom - 1, int_8)*INT(natom, int_8) + INT(jatom, int_8)
!$       hash1 = INT(MOD(iatom8, INT(nlock, int_8)) + 1)

         ! basis ikind
         first_sgfa => basis_set_a%first_sgf
         la_max => basis_set_a%lmax
         la_min => basis_set_a%lmin
         npgfa => basis_set_a%npgf
         nseta = basis_set_a%nset
         nsgfa => basis_set_a%nsgf_set
         rpgfa => basis_set_a%pgf_radius
         set_radius_a => basis_set_a%set_radius
         scon_a => basis_set_a%scon
         zeta => basis_set_a%zet
         ! basis jkind
         first_sgfb => basis_set_b%first_sgf
         lb_max => basis_set_b%lmax
         lb_min => basis_set_b%lmin
         npgfb => basis_set_b%npgf
         nsetb = basis_set_b%nset
         nsgfb => basis_set_b%nsgf_set
         rpgfb => basis_set_b%pgf_radius
         set_radius_b => basis_set_b%set_radius
         scon_b => basis_set_b%scon
         zetb => basis_set_b%zet

         IF (use_cell_mapping) THEN
            ic = cell_to_index(cell(1), cell(2), cell(3))
            CPASSERT(ic > 0)
         ELSE
            ic = 1
         END IF

         IF (do_symmetric) THEN
            IF (iatom <= jatom) THEN
               irow = iatom
               icol = jatom
            ELSE
               irow = jatom
               icol = iatom
            END IF
            f0 = 2.0_dp
            ff = 2.0_dp
            IF (iatom == jatom) f0 = 1.0_dp
         ELSE
            irow = iatom
            icol = jatom
            f0 = 1.0_dp
            ff = 1.0_dp
         END IF
         DO i = 1, maxs
            NULLIFY (sint(i)%block)
            IF (dokp) THEN
               CALL dbcsr_get_block_p(matrix=matrixkp_s(i, ic)%matrix, &
                                      row=irow, col=icol, BLOCK=sint(i)%block, found=found)
               CPASSERT(found)
            ELSE
               CALL dbcsr_get_block_p(matrix=matrix_s(i)%matrix, &
                                      row=irow, col=icol, BLOCK=sint(i)%block, found=found)
               CPASSERT(found)
            END IF
         END DO
         IF (do_forces) THEN
            NULLIFY (p_block)
            IF (dokp) THEN
               CALL dbcsr_get_block_p(matrix=matrixkp_p(1, ic)%matrix, &
                                      row=irow, col=icol, block=p_block, found=found)
               CPASSERT(found)
            ELSE
               CALL dbcsr_get_block_p(matrix=matrix_p, row=irow, col=icol, &
                                      block=p_block, found=found)
               CPASSERT(found)
            END IF
         END IF
         trans = do_symmetric .AND. (iatom > jatom)

         rab2 = rab(1)*rab(1) + rab(2)*rab(2) + rab(3)*rab(3)
         dab = SQRT(rab2)

         DO iset = 1, nseta

            ncoa = npgfa(iset)*ncoset(la_max(iset))
            n1 = npgfa(iset)*(ncoset(la_max(iset)) - ncoset(la_min(iset) - 1))
            sgfa = first_sgfa(1, iset)

            DO jset = 1, nsetb

               IF (set_radius_a(iset) + set_radius_b(jset) < dab) CYCLE

!$             hash2 = MOD((iset - 1)*nsetb + jset, nlock) + 1
!$             hash = MOD(hash1 + hash2, nlock) + 1

               ncob = npgfb(jset)*ncoset(lb_max(jset))
               n2 = npgfb(jset)*(ncoset(lb_max(jset)) - ncoset(lb_min(jset) - 1))
               sgfb = first_sgfb(1, jset)


!               print *, slot, ikind, jkind, iatom, jatom, iset, jset
!               print *, " ----> ",  shape(sint(1)%block), nsgfa(iset), nsgfb(jset), sgfa, sgfb, trans, maxs

               ! calculate integrals and derivatives
               SELECT CASE (nder)
               CASE (0)
                  CALL overlap_ab(la_max(iset), la_min(iset), npgfa(iset), rpgfa(:, iset), zeta(:, iset), &
                                  lb_max(jset), lb_min(jset), npgfb(jset), rpgfb(:, jset), zetb(:, jset), &
                                  rab, sab=oint(:, :, 1))
               CASE (1)
                  CALL overlap_ab(la_max(iset), la_min(iset), npgfa(iset), rpgfa(:, iset), zeta(:, iset), &
                                  lb_max(jset), lb_min(jset), npgfb(jset), rpgfb(:, jset), zetb(:, jset), &
                                  rab, sab=oint(:, :, 1), dab=oint(:, :, 2:4))
               CASE (2)
                  CALL overlap_ab(la_max(iset), la_min(iset), npgfa(iset), rpgfa(:, iset), zeta(:, iset), &
                                  lb_max(jset), lb_min(jset), npgfb(jset), rpgfb(:, jset), zetb(:, jset), &
                                  rab, sab=oint(:, :, 1), dab=oint(:, :, 2:4), ddab=oint(:, :, 5:10))
               CASE DEFAULT
                  CPABORT("")
               END SELECT
               IF (do_forces .AND. ASSOCIATED(p_block) .AND. ((iatom /= jatom) .OR. use_virial)) THEN
                  ! Decontract P matrix block
                  owork = 0.0_dp
                  CALL block_add("OUT", owork, nsgfa(iset), nsgfb(jset), p_block, sgfa, sgfb, trans=trans)
                  CALL decontraction(owork, pmat, scon_a(:, sgfa:), n1, nsgfa(iset), scon_b(:, sgfb:), n2, &
                                     nsgfb(jset), trans=trans)
                  CALL force_trace(force_a, oint(:, :, 2:4), pmat, n1, n2, 3)
                  force_thread(:, iatom) = force_thread(:, iatom) - ff*force_a(:)
                  force_thread(:, jatom) = force_thread(:, jatom) + ff*force_a(:)
                  IF (use_virial) THEN
                     CALL virial_pair_force(pv_thread, -f0, force_a, rab)
                  END IF
               END IF
               ! Contraction
               DO i = 1, maxs
                  f = 1.0_dp
                  IF (ndod(i) == 1 .AND. trans) f = -1.0_dp
                  CALL contraction(oint(:, :, i), owork, ca=scon_a(:, sgfa:), na=n1, ma=nsgfa(iset), &
                                   cb=scon_b(:, sgfb:), nb=n2, mb=nsgfb(jset), fscale=f, trans=trans)
!$                CALL omp_set_lock(locks(hash))
                  CALL block_add("IN", owork, nsgfa(iset), nsgfb(jset), sint(i)%block, &
                                 sgfa, sgfb, trans=trans)
                  ma = nsgfa(iset)
                  mb = nsgfb(jset)
                  if (trans) then
                     temp_qab_matrix_ref = owork(1:mb, 1:ma)
                  else
                     temp_qab_matrix_ref = owork(1:ma, 1:mb)
                  end if
!                  print *, 'Ref', slot, iset, jset, ma, mb, trans, temp_qab_matrix_ref
!$                CALL omp_unset_lock(locks(hash))
               END DO

            END DO
         END DO
      END DO
      IF (do_forces) DEALLOCATE (pmat)
      DEALLOCATE (oint, owork)
      DEALLOCATE (sint)
!$OMP DO
!$    DO lock_num = 1, nlock
!$       call omp_destroy_lock(locks(lock_num))
!$    END DO
!$OMP END DO

!$OMP SINGLE
!$    DEALLOCATE (locks)
!$OMP END SINGLE NOWAIT

!$OMP END PARALLEL

      IF (do_forces) THEN
         ALLOCATE (atom_of_kind(natom), kind_of(natom))
         CALL get_atomic_kind_set(atomic_kind_set, atom_of_kind=atom_of_kind, &
                                  kind_of=kind_of)
!$OMP DO
         DO iatom = 1, natom
            atom_a = atom_of_kind(iatom)
            ikind = kind_of(iatom)
            force(ikind)%overlap(:, atom_a) = force(ikind)%overlap(:, atom_a) + force_thread(:, iatom)
         END DO
!$OMP END DO
         DEALLOCATE (atom_of_kind, kind_of)
      END IF
      IF (do_forces .AND. use_virial) THEN
         virial%pv_overlap = virial%pv_overlap + pv_thread
         virial%pv_virial = virial%pv_virial + pv_thread
      END IF

      IF (dokp) THEN
         DO i = 1, maxs
            DO ic = 1, nimg
               CALL dbcsr_finalize(matrixkp_s(i, ic)%matrix)
               CALL dbcsr_filter(matrixkp_s(i, ic)%matrix, &
                                 dft_control%qs_control%eps_filter_matrix)
            END DO
         END DO
      ELSE
         DO i = 1, maxs
            CALL dbcsr_finalize(matrix_s(i)%matrix)
            CALL dbcsr_filter(matrix_s(i)%matrix, &
                              dft_control%qs_control%eps_filter_matrix)
         END DO
      END IF


!!!!!!!!!! starts here !!!!!!!!!!!!
      print *, "Intercepted call to build overlap ", sab_nl(1)%nl_size, ldsab, ldsab, maxder,&
                 do_forces, do_symmetric, use_cell_mapping, dokp, nkind, size(basis_set_list_a), size(basis_set_list_b), natom

      ! precompute env size
      offset = 21 ! skip the first 20 values, like libCint
      allocate( env(offset+100) )
      env = 0_dp

      call gen_libcint_env( env, atm_a, bas_a, offset, basis_set_list_a, atomic_kind_set, natom)
      call gen_libcint_env( env, atm_b, bas_b, offset, basis_set_list_b, atomic_kind_set, natom)

      print *, 'Atm_a:'
      do iatom=1, natom
         print *, atm_a(:, iatom)
      end do
      print *, 'Bas_a:'
      do iset=1, size(bas_a, 2)
         print *, bas_a(:, iset)
      end do
!      print *, 'Atm_b:', atm_b
!      print *, 'Bas_b:', bas_b


!      call gen_sab_nl_set( sab_nl_set, sab_nl, basis_set_a, basis_set_b, atm_a, atm_b, sab, qab )
      nsetab = 0
      offset_to_add = 1
      DO slot = 1, sab_nl(1)%nl_size
         ikind = sab_nl(1)%nlist_task(slot)%ikind
         jkind = sab_nl(1)%nlist_task(slot)%jkind
         basis_set_a => basis_set_list_a(ikind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_a)) cycle
         basis_set_b => basis_set_list_b(jkind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_b)) cycle
         nseta = basis_set_a%nset
         nsetb = basis_set_b%nset
         nsetab = nsetab + nseta*nsetb
         set_radius_a => basis_set_a%set_radius
         set_radius_b => basis_set_b%set_radius

         DO iset = 1, nseta
            DO jset = 1, nsetb
               if (set_radius_a(iset) + set_radius_b(jset) < dab) then
                  cycle
               else
                  offset_to_add = offset_to_add + 3
               end if
            end do
         end do
                  
      end do
      allocate( sab_nl_set(8,nsetab) )

      old_env = env
      deallocate(env)
      allocate( env(offset+offset_to_add) )
      print *, 'Offset:', offset, '->', offset+offset_to_add
      env(:offset) = old_env(:offset)

      sab_dest = 1
      qab_dest = 1
      set_slot = 1
      
      DO slot = 1, sab_nl(1)%nl_size
         ikind = sab_nl(1)%nlist_task(slot)%ikind
         jkind = sab_nl(1)%nlist_task(slot)%jkind
         iatom = sab_nl(1)%nlist_task(slot)%iatom
         jatom = sab_nl(1)%nlist_task(slot)%jatom
         cell(:) = sab_nl(1)%nlist_task(slot)%cell(:)
         rab(1:3) = sab_nl(1)%nlist_task(slot)%r(1:3)
         rab2 = rab(1)*rab(1) + rab(2)*rab(2) + rab(3)*rab(3)
         dab = SQRT(rab2)
         basis_set_a => basis_set_list_a(ikind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_a)) cycle
         basis_set_b => basis_set_list_b(jkind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_b)) cycle
         !
         nsgfa => basis_set_a%nsgf_set
         nsgfb => basis_set_b%nsgf_set
         npgfa => basis_set_a%npgf
         npgfb => basis_set_b%npgf
         set_radius_a => basis_set_a%set_radius
         set_radius_b => basis_set_b%set_radius
         la_max => basis_set_a%lmax
         lb_max => basis_set_b%lmax
         la_min => basis_set_a%lmin
         lb_min => basis_set_b%lmin
         nseta = basis_set_a%nset
         nsetb = basis_set_b%nset
!         call compute_ic( ic, use_cell_mapping, cell_to_index, cell )
!         call apply_simmetry( do_symmetric, iatom, jatom, irow, icol, f0, ff )
!         call absurdly_complicate_get_block( way too many args  )

         DO iset = 1, nseta
            DO jset = 1, nsetb
               IF (set_radius_a(iset) + set_radius_b(jset) < dab) cycle
!               print *, slot, ikind, jkind, iatom, jatom, iset, jset, cell, rab, set_slot, curr_dest
               ! for each pair of set, save:
               ! 1) the index in sab_nl
               ! 2) the index in bas_a with the information for the a set
               ! 3) same for b
               ! 4) where to save sab
               ! 5) where to save qab
               ! 6 & 7) debug, index of set in the sab_nl
               ! 8) offset to env with the distance between centers
               n1 = npgfa(iset)*(ncoset(la_max(iset)) - ncoset(la_min(iset) - 1))
               n2 = npgfb(jset)*(ncoset(lb_max(jset)) - ncoset(lb_min(jset) - 1))
!               index_of_starting_set_ai = atm_a(5,iatom)
!               index_of_starting_set_bj = atm_b(5,jatom)
               ! sab_nl goes over atoms pairs. Different atom pairs have
               ! different nset pairs.
               ! Reconstruct where in bas the information about these sets is
               ! stored.
               offset_ai = atm_a(5,iatom)
               offset_bj = atm_b(5,jatom)
               sab_nl_set( :, set_slot ) = [ slot, offset_ai+iset-1, offset_bj+jset-1, sab_dest, qab_dest, iset, jset, offset+1 ]
               env ( offset+1 ) = sab_nl(1)%nlist_task(slot)%r(1)
               env ( offset+2 ) = sab_nl(1)%nlist_task(slot)%r(2)
               env ( offset+3 ) = sab_nl(1)%nlist_task(slot)%r(3)
               !
               qab_dest = qab_dest + nsgfa(iset) * nsgfb(jset)
               sab_dest = sab_dest + n1 * n2
               set_slot = set_slot + 1
               offset = offset + 3
            end do
         end do
      end do
      nsetab = set_slot - 1
      allocate(temp_sab(sab_dest-1), temp_qab(qab_dest-1))
!      end gen_sab_nl_set
!      call compute_qab_from_nlset( sab_nl_set, bas_a, bas_b, temp_sab, env, do_symmetric )
      do set_slot = 1, nsetab
         !
         slot = sab_nl_set( 1, set_slot )
         iset = sab_nl_set( 2, set_slot )
         jset = sab_nl_set( 3, set_slot )
         sab_dest = sab_nl_set( 4, set_slot )
         qab_dest = sab_nl_set( 5, set_slot )
         !
         r_offset = sab_nl_set( 8, set_slot )
         rab(1:3) = env( r_offset: r_offset+3-1 )
         ! a
         iatom = bas_a(1, iset)
         la_max_iset = bas_a(2, iset)
         la_min_iset = bas_a(8, iset)
         npgfa_iset  = bas_a(3, iset)
         nsgfa_iset  = bas_a(4, iset)
         ! unused spinor kappa = bas_a(5, iset)
         z_offseta = bas_a(6, iset)
         c_offseta = bas_a(7, iset)       
         ma = nsgfa_iset
         n1 = npgfa_iset*(ncoset(la_max_iset) - ncoset(la_min_iset-1))
         zeta_iset   = env( z_offseta: z_offseta+npgfa_iset-1 )
         rpgfa_iset  = env( z_offseta+npgfa_iset: z_offseta+2*npgfa_iset-1 )
         scon_a_iset = env( c_offseta: c_offseta+n1*ma-1 )
         ! b
         jatom = bas_b(1, jset)
         lb_max_jset = bas_b(2, jset)
         lb_min_jset = bas_b(8, jset)
         npgfb_jset  = bas_b(3, jset)
         nsgfb_jset  = bas_b(4, jset)
         z_offsetb = bas_b(6, jset)
         c_offsetb = bas_b(7, jset)
         mb = nsgfb_jset
         n2 = npgfb_jset*(ncoset(lb_max_jset) - ncoset(lb_min_jset-1))
         zetb_jset   = env( z_offsetb: z_offsetb+npgfb_jset-1 )
         rpgfb_jset  = env( z_offsetb+npgfb_jset: z_offsetb+2*npgfb_jset-1 )
         scon_b_jset = env( c_offsetb: c_offsetb+n2*mb-1 )
         ! trans
         trans = do_symmetric .AND. (iatom > jatom)
         ! S = <set_a|set_b> 
         temp_sab_matrix = reshape( temp_sab(sab_dest:sab_dest+n1*n2-1), (/ n2, n1 /) )
         ! matrices and vectors are actually different object in fortran. Weird
         temp_sab_matrix_cpu = reshape( temp_sab(sab_dest:sab_dest+n1*n2-1), (/ n1, n2 /) )
!       subroutine overlap_ab_gpu( la_max, la_min, npgfa, irpgfa, zeta, lb_max, lb_min, npgfb, rpgfb, zetb, rab, sab, lds ) &
         !
         CALL overlap_ab_gpu(la_max_iset, la_min_iset, npgfa_iset, c_loc(rpgfa_iset(1)), c_loc(zeta_iset(1)), &
                             lb_max_jset, lb_min_jset, npgfb_jset, c_loc(rpgfb_jset(1)), c_loc(zetb_jset(1)), &
                             rab=c_loc(rab(1)), sab=c_loc(temp_sab_matrix(1,1)), lds=n2)
         ! compare against cpu calculation with the same input
         CALL overlap_ab(la_max = la_max_iset, la_min = la_min_iset, npgfa=npgfa_iset, &
                             rpgfa=rpgfa_iset, zeta=zeta_iset, &
                             lb_max = lb_max_jset, lb_min = lb_min_jset, npgfb=npgfb_jset, &
                             rpgfb=rpgfb_jset, zetb=zetb_jset, &
                             rab=rab, sab=temp_sab_matrix_cpu)
         !
         temp_sab_matrix = transpose(temp_sab_matrix)

         if (sum(abs( temp_sab_matrix-temp_sab_matrix_cpu)) > 1.0E-10_dp ) then
            print *, 'Check 1 Diff', sum(abs( temp_sab_matrix-temp_sab_matrix_cpu))
            iset_iset = sab_nl_set( 6, set_slot )
            jset_jset = sab_nl_set( 7, set_slot )
            print *, set_slot, iset, jset, iset_iset, jset_jset, la_max_iset, la_min_iset, lb_max_jset, lb_min_jset, n1, n2
            print *, 'GPU'
            do i=1, n1
               print *, temp_sab_matrix(i,:)
            end do
            print *, 'CPU'
            do i=1, n1
               print *, temp_sab_matrix_cpu(i,:)
            end do
            print *, '---------------------------------'
         end if

         ! compare against the results from the legacy routine called using the data from sab_nl
         ikind = sab_nl(1)%nlist_task(slot)%ikind
         jkind = sab_nl(1)%nlist_task(slot)%jkind
         rab(1:3) = sab_nl(1)%nlist_task(slot)%r(1:3)
         !
         basis_set_a => basis_set_list_a(ikind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_a)) cycle
         basis_set_b => basis_set_list_b(jkind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_b)) cycle
         !
         ! basis ikind
         first_sgfa => basis_set_a%first_sgf
         la_max => basis_set_a%lmax
         la_min => basis_set_a%lmin
         npgfa => basis_set_a%npgf
         nseta = basis_set_a%nset
         nsgfa => basis_set_a%nsgf_set
         rpgfa => basis_set_a%pgf_radius
         set_radius_a => basis_set_a%set_radius
         scon_a => basis_set_a%scon
         zeta => basis_set_a%zet
         ! basis jkind
         first_sgfb => basis_set_b%first_sgf
         lb_max => basis_set_b%lmax
         lb_min => basis_set_b%lmin
         npgfb => basis_set_b%npgf
         nsetb = basis_set_b%nset
         nsgfb => basis_set_b%nsgf_set
         rpgfb => basis_set_b%pgf_radius
         set_radius_b => basis_set_b%set_radius
         scon_b => basis_set_b%scon
         zetb => basis_set_b%zet
         !
         iset_iset = sab_nl_set( 6, set_slot ) 
         ncoa = npgfa(iset_iset)*ncoset(la_max(iset_iset))
         n1 = npgfa(iset_iset)*(ncoset(la_max(iset_iset)) - ncoset(la_min(iset_iset)-1))
         sgfa = first_sgfa(1, iset_iset)  
         jset_jset = sab_nl_set( 7, set_slot )
         ncob = npgfb(jset_jset)*ncoset(lb_max(jset_jset))
         n2 = npgfb(jset_jset)*(ncoset(lb_max(jset_jset)) - ncoset(lb_min(jset_jset)-1))
         sgfb = first_sgfb(1, jset_jset)

         ! calculate integrals and derivatives
         CALL overlap_ab(la_max(iset_iset), la_min(iset_iset), npgfa(iset_iset), &
                         rpgfa(:, iset_iset), zeta(:, iset_iset), &
                         lb_max(jset_jset), lb_min(jset_jset), npgfb(jset_jset), &
                         rpgfb(:, jset_jset), zetb(:, jset_jset), &
                         rab, sab=temp_sab_matrix_cpu)
         if (sum(abs( temp_sab_matrix-temp_sab_matrix_cpu)) > 1.0E-10_dp ) then
            print *, 'Check 2 Diff', sum(abs( temp_sab_matrix-temp_sab_matrix_cpu))
            iset_iset = sab_nl_set( 6, set_slot )
            jset_jset = sab_nl_set( 7, set_slot )
            print *, set_slot, iset, jset, iset_iset, jset_jset, la_max_iset, la_min_iset, lb_max_jset, lb_min_jset, n1, n2
            print *, 'GPU'
            do i=1, n1
               print *, temp_sab_matrix(i,:)
            end do
            print *, 'CPU'
            do i=1, n1
               print *, temp_sab_matrix_cpu(i,:)
            end do
            print *, '---------------------------------'
         end if
         !
         ! Q = ( Ca.T @ ( S Cb )) ^ trans
         ca = reshape( scon_a_iset, (/ n1, ma /) )
         cb = reshape( scon_b_jset, (/ n2, mb /) )
         if (trans) then
            temp_qab_matrix = reshape( temp_qab(qab_dest:qab_dest+ma*mb-1), (/ mb, ma /) )
         else
            temp_qab_matrix = reshape( temp_qab(qab_dest:qab_dest+ma*mb-1), (/ ma, mb /) )
         end if

         ! fscale=1.0_dp when doing overlap. It will change when derivatives are
         ! included
         CALL contraction(temp_sab_matrix, temp_qab_matrix, ca=ca, na=n1, ma=ma, &
                          cb=cb, nb=n2, mb=mb, fscale=1.0_dp, trans=trans)
!         rab2 = rab(1)*rab(1) + rab(2)*rab(2) + rab(3)*rab(3)
!         dab = SQRT(rab2)

!         print *, temp_qab_matrix_ref - temp_qab_matrix
!         deallocate(sint)
         deallocate(temp_sab_matrix, temp_qab_matrix, ca, cb)
      end do
!      end compute_qab_from_nlset
!      DEALLOCATE (sint)

      ! *** Release work storage ***
      DEALLOCATE (basis_set_list_a, basis_set_list_b)

      CALL timestop(handle)

   END SUBROUTINE build_overlap_matrix_low

! Fill the env, atm and bas arrays with information from basis_set_list and atomic_kind_set.
! see https://github.com/sunqm/libcint/blob/master/doc/program_ref.pdf for a
! description of the output arrays. Note that:
! change a) we add the rpfg values in the env array immediately after the zet values
! change b) tech. speaking, the bas array contains sets, not shells. A set is a
! group of shells with the same zeta. A set can (and often does) contain many shells 
! of different moments. A set can contain many shells with the same moment
! change c) SETS BELONGING TO THE SAME ATOM KIND MUST BE ONE AFTER THE OTHER IN BAS !
! change d) atm(5,:) contains the (address in bas to the) first set in the atom kind
subroutine gen_libcint_env( env, atm, bas, offset, basis_set_list, atomic_kind_set, natom)
   implicit none
   real(kind=dp), allocatable, dimension(:), intent(inout) :: env
   integer, allocatable, dimension(:,:), intent(out) :: atm
   integer, allocatable, dimension(:,:), intent(out) :: bas
   integer, intent(inout) :: offset
   TYPE(gto_basis_set_p_type), DIMENSION(:), POINTER  :: basis_set_list
   TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
   integer, intent(in) :: natom

   integer :: max_nset, offset_to_add, ikind, nset, sgf, n, m, k
   integer :: bas_size, iset, iatom, ibas
   TYPE(gto_basis_set_type), POINTER                  :: basis_set
   INTEGER, DIMENSION(:), POINTER                     :: lmax, lmin, npgf, nsgf
   INTEGER, DIMENSION(:, :), POINTER                  :: first_sgf
   integer, allocatable, dimension(:,:)               :: z_offset, c_offset
   integer, allocatable, dimension(:)                 :: atom_of_kind, kind_of
   REAL(kind=dp), DIMENSION(:, :), POINTER            :: rpgf, scon, zet
   real(kind=dp), allocatable, dimension(:)           :: old_env

   ! 1) precompute size of env, bas, atm and the temporary arrays
   max_nset = 0
   offset_to_add = 0 ! todo or is it 1 ?
   do ikind=1, size(basis_set_list)
      basis_set => basis_set_list(ikind)%gto_basis_set
      lmax => basis_set%lmax
      lmin => basis_set%lmin
      nset =  basis_set%nset
      npgf => basis_set%npgf
      nsgf => basis_set%nsgf_set
      max_nset = max(max_nset,nset)
      do iset=1, nset
         n = npgf(iset)*(ncoset(lmax(iset)) - ncoset(lmin(iset) - 1)) ! number of rows in the co (sub)matrix
         m = nsgf(iset) ! number of columns in the co sub(matrix)
         k = npgf(iset) ! number of gaussian exponents ( and number of lenghts rpgf )
         offset_to_add = offset_to_add + n*m + k*2
      end do
   end do
   ALLOCATE (atom_of_kind(natom), kind_of(natom))
   CALL get_atomic_kind_set(atomic_kind_set, atom_of_kind=atom_of_kind, kind_of=kind_of)
   bas_size = 0
   do iatom=1, natom
      ikind = kind_of(iatom)
      basis_set => basis_set_list(ikind)%gto_basis_set ! todo check if ikind in basis set
      nset = basis_set%nset
      bas_size = bas_size + nset
   end do 

   ! 2) Allocate memory, copy old env in the new env
!   allocate(old_env(offset-1))
   old_env = env
   deallocate(env)
   allocate( env(offset+offset_to_add) )
   print *, 'Offset:', offset, '->', offset+offset_to_add
   env(:offset) = old_env(:offset)
   allocate( z_offset(size(basis_set_list),max_nset) )
   allocate( c_offset(size(basis_set_list),max_nset) )

   allocate( bas(8,bas_size) ) ! todo define 8 and 6
   allocate( atm(6,natom) )
   ! 3) fill the env and offset arrays
   do ikind=1, size(basis_set_list)
      basis_set => basis_set_list(ikind)%gto_basis_set
!      print *, "SGF",  basis_set%first_sgf(1,:)
!      print *, "LM",   basis_set%lmax
!      print *, "Lm",   basis_set%lmin
!      print *, "NPGF", basis_set%npgf
!      print *, "NSET", basis_set%nset
!      print *, "NSGF", basis_set%nsgf_set
!      print *, "shape of rpgf ", shape(basis_set%pgf_radius), "set_radius", shape(basis_set%set_radius)
!      print *, "shape of scon", shape(basis_set%scon)      , "zet", shape(basis_set%zet)
      lmax => basis_set%lmax
      lmin => basis_set%lmin
      nset =  basis_set%nset
      npgf => basis_set%npgf
      nsgf => basis_set%nsgf_set
      rpgf => basis_set%pgf_radius
      scon => basis_set%scon
      first_sgf => basis_set%first_sgf
      zet  => basis_set%zet
      scon => basis_set%scon
      do iset=1, nset
         n = npgf(iset)*(ncoset(lmax(iset)) - ncoset(lmin(iset) - 1)) ! number of rows in the co (sub)matrix
         m = nsgf(iset) ! number of columns in the co sub(matrix)
         k = npgf(iset) ! number of gaussian exponents ( and number of lenghts rpgf )
         sgf = first_sgf(1, iset) ! starting col of the co submatrix in the scon matrix
         z_offset(ikind,iset) = offset
         env( offset:offset+k-1 ) = zet(:,iset)
         env( offset+k:offset+2*k-1 ) = rpgf(:,iset)
         offset = offset + k*2 ! zet and r one after the other
         c_offset(ikind,iset) = offset
         ! scon is a ~list matrix with the contraction coefficient already
         ! premultiplied with the cartesian2spherical matrix.
         ! The number of row of scon is the max n, the number of col is the sum
         ! of all sgf. Note that this mean a generic scon submatrix is not
         ! countinous -_- so we make it
         env( offset:offset+n*m-1 ) = reshape( scon(:n,sgf:sgf+m-1), (/ n*m /) )
         offset = offset + n*m
!         print *, ikind, iset, npgf(iset), n, m, k, sgf, offset
!         print *, env( z_offset(ikind,iset) ), env( c_offset(ikind,iset) )
      end do
   end do
   ! 4) (assuming all atoms are in basis_set_list) fill the bas array with the
   ! basis set, and fill the atm array
   ibas = 1
   do iatom=1, natom
!      atom_a = atom_of_kind(iatom)
      ikind = kind_of(iatom)
!      print *, "--------", iatom, atom_a, ikind ! index, atom index in same kind sublist, kind of atom
      basis_set => basis_set_list(ikind)%gto_basis_set ! todo check if ikind in basis set
      nset = basis_set%nset
      lmax => basis_set%lmax
      lmin => basis_set%lmin
      npgf => basis_set%npgf
      nsgf => basis_set%nsgf_set
!      set_radius => basis_set%set_radius
!     1) Z
!     2) offset to R
!     3) Nuclear model
!     4) offset to zeta_nuclear
!     5) start of this atom basis sets in the bas array
!     6) unused
      atm(:,iatom) = [ 0, 0, 0, 0, ibas, 0 ] ! TODO add r
      do iset=1, nset
!         print *, "  --->    ", iatom, lmax(iset), npgf(iset), nsgf(iset), &
!                  0, z_offset(ikind,iset), c_offset(ikind,iset), lmin(iset)
!         print *, ibas, '/', bas_size, shape(bas)
         bas(:,ibas) = [ iatom, lmax(iset), npgf(iset), nsgf(iset), 0, z_offset(ikind,iset), c_offset(ikind,iset), lmin(iset) ]
         ibas = ibas + 1
      end do 
   end do
   ! 5) bye
   DEALLOCATE (atom_of_kind, kind_of)
   deallocate( old_env )
   deallocate( c_offset)
   deallocate( z_offset)


end subroutine gen_libcint_env



! **************************************************************************************************
!> \brief   Calculation of the overlap matrix over Cartesian Gaussian functions.
!> \param   ks_env the QS env
!> \param   matrix_s The overlap matrix to be calculated
!> \param   basis_set_list_a basis set list to be used for bra in <a|b>
!> \param   basis_set_list_b basis set list to be used for ket in <a|b>
!> \param   sab_nl pair list (must be consistent with basis sets!)
!> \date    11.03.2016
!> \par     History
!>          Simplified version of build_overlap_matrix
!> \author  MK
!> \version 1.0
! **************************************************************************************************
   SUBROUTINE build_overlap_matrix_simple(ks_env, matrix_s, &
                                          basis_set_list_a, basis_set_list_b, sab_nl)

      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: matrix_s
      TYPE(gto_basis_set_p_type), DIMENSION(:), POINTER  :: basis_set_list_a, basis_set_list_b
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: sab_nl

      CHARACTER(len=*), PARAMETER :: routineN = 'build_overlap_matrix_simple'

      INTEGER                                            :: handle, iatom, icol, ikind, irow, iset, &
                                                            jatom, jkind, jset, ldsab, m1, m2, n1, &
                                                            n2, natom, ncoa, ncob, nkind, nseta, &
                                                            nsetb, sgfa, sgfb, slot
      INTEGER, DIMENSION(:), POINTER                     :: la_max, la_min, lb_max, lb_min, npgfa, &
                                                            npgfb, nsgfa, nsgfb
      INTEGER, DIMENSION(:, :), POINTER                  :: first_sgfa, first_sgfb
      LOGICAL                                            :: do_symmetric, found, trans
      REAL(KIND=dp)                                      :: dab, rab2
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: owork
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :)     :: oint
      REAL(KIND=dp), DIMENSION(3)                        :: rab
      REAL(KIND=dp), DIMENSION(:), POINTER               :: set_radius_a, set_radius_b
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: rpgfa, rpgfb, scon_a, scon_b, zeta, zetb
      TYPE(atomic_kind_type), DIMENSION(:), POINTER      :: atomic_kind_set
      TYPE(block_p_type), ALLOCATABLE, DIMENSION(:)      :: sint
      TYPE(dft_control_type), POINTER                    :: dft_control
      TYPE(gto_basis_set_type), POINTER                  :: basis_set_a, basis_set_b
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set

!$    INTEGER(kind=omp_lock_kind), &
!$       ALLOCATABLE, DIMENSION(:) :: locks
!$    INTEGER                                            :: lock_num, hash, hash1, hash2
!$    INTEGER(KIND=int_8)                                :: iatom8
!$    INTEGER, PARAMETER                                 :: nlock = 501

      NULLIFY (dft_control)

      CALL timeset(routineN, handle)

      NULLIFY (atomic_kind_set)
      CALL get_ks_env(ks_env, &
                      atomic_kind_set=atomic_kind_set, &
                      natom=natom, &
                      qs_kind_set=qs_kind_set, &
                      dft_control=dft_control)

      ! check for symmetry
      CPASSERT(SIZE(sab_nl) > 0)
      CALL get_neighbor_list_set_p(neighbor_list_sets=sab_nl, symmetric=do_symmetric)

      nkind = SIZE(qs_kind_set)

      CALL dbcsr_allocate_matrix_set(matrix_s, 1)
      CALL create_sab_matrix(ks_env, matrix_s, "Matrix", basis_set_list_a, basis_set_list_b, &
                             sab_nl, do_symmetric)

      ldsab = 0
      DO ikind = 1, nkind
         basis_set_a => basis_set_list_a(ikind)%gto_basis_set
         CALL get_gto_basis_set(gto_basis_set=basis_set_a, maxco=m1, nsgf=m2)
         ldsab = MAX(m1, m2, ldsab)
         basis_set_b => basis_set_list_b(ikind)%gto_basis_set
         CALL get_gto_basis_set(gto_basis_set=basis_set_b, maxco=m1, nsgf=m2)
         ldsab = MAX(m1, m2, ldsab)
      END DO

!$OMP PARALLEL DEFAULT(NONE) &
!$OMP SHARED (ldsab,sab_nl,do_symmetric,ncoset,natom,&
!$OMP         matrix_s,basis_set_list_a,basis_set_list_b,locks)  &
!$OMP PRIVATE (oint,owork,sint,ikind,jkind,iatom,jatom,rab,basis_set_a,basis_set_b,&
!$OMP          first_sgfa, la_max, la_min, npgfa, nsgfa, nseta, rpgfa, set_radius_a, ncoa, ncob, &
!$OMP          zeta, first_sgfb, lb_max, lb_min, npgfb, nsetb, rpgfb, set_radius_b, nsgfb, dab, &
!$OMP          zetb, scon_a, scon_b, irow, icol, found, trans, rab2, n1, n2, sgfa, sgfb, iset, jset, &
!$OMP          slot, lock_num, hash, hash1, hash2, iatom8 )

!$OMP SINGLE
!$    ALLOCATE (locks(nlock))
!$OMP END SINGLE

!$OMP DO
!$    DO lock_num = 1, nlock
!$       call omp_init_lock(locks(lock_num))
!$    END DO
!$OMP END DO

      ALLOCATE (oint(ldsab, ldsab, 1), owork(ldsab, ldsab))
      ALLOCATE (sint(1))
      NULLIFY (sint(1)%block)

      print *, "Intercepted call to build overlap simple ", sab_nl(1)%nl_size, ldsab, ldsab, do_symmetric
!$OMP DO SCHEDULE(GUIDED)
      DO slot = 1, sab_nl(1)%nl_size
         ikind = sab_nl(1)%nlist_task(slot)%ikind
         jkind = sab_nl(1)%nlist_task(slot)%jkind
         iatom = sab_nl(1)%nlist_task(slot)%iatom
         jatom = sab_nl(1)%nlist_task(slot)%jatom
         rab(1:3) = sab_nl(1)%nlist_task(slot)%r(1:3)
!$       iatom8 = (iatom - 1)*natom + jatom
!$       hash1 = INT(MOD(iatom8, INT(nlock, int_8)) + 1)

         basis_set_a => basis_set_list_a(ikind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_a)) CYCLE
         basis_set_b => basis_set_list_b(jkind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_b)) CYCLE
         ! basis ikind
         first_sgfa => basis_set_a%first_sgf
         la_max => basis_set_a%lmax
         la_min => basis_set_a%lmin
         npgfa => basis_set_a%npgf
         nseta = basis_set_a%nset
         nsgfa => basis_set_a%nsgf_set
         rpgfa => basis_set_a%pgf_radius
         set_radius_a => basis_set_a%set_radius
         scon_a => basis_set_a%scon
         zeta => basis_set_a%zet
         ! basis jkind
         first_sgfb => basis_set_b%first_sgf
         lb_max => basis_set_b%lmax
         lb_min => basis_set_b%lmin
         npgfb => basis_set_b%npgf
         nsetb = basis_set_b%nset
         nsgfb => basis_set_b%nsgf_set
         rpgfb => basis_set_b%pgf_radius
         set_radius_b => basis_set_b%set_radius
         scon_b => basis_set_b%scon
         zetb => basis_set_b%zet

         IF (do_symmetric) THEN
            IF (iatom <= jatom) THEN
               irow = iatom
               icol = jatom
            ELSE
               irow = jatom
               icol = iatom
            END IF
         ELSE
            irow = iatom
            icol = jatom
         END IF

         NULLIFY (sint(1)%block)
         CALL dbcsr_get_block_p(matrix=matrix_s(1)%matrix, &
                                row=irow, col=icol, BLOCK=sint(1)%block, found=found)
         CPASSERT(found)
         trans = do_symmetric .AND. (iatom > jatom)

         rab2 = rab(1)*rab(1) + rab(2)*rab(2) + rab(3)*rab(3)
         dab = SQRT(rab2)

         DO iset = 1, nseta

            ncoa = npgfa(iset)*ncoset(la_max(iset))
            n1 = npgfa(iset)*(ncoset(la_max(iset)) - ncoset(la_min(iset) - 1))
            sgfa = first_sgfa(1, iset)

            DO jset = 1, nsetb

               IF (set_radius_a(iset) + set_radius_b(jset) < dab) CYCLE

!$             hash2 = MOD((iset - 1)*nsetb + jset, nlock) + 1
!$             hash = MOD(hash1 + hash2, nlock) + 1

               ncob = npgfb(jset)*ncoset(lb_max(jset))
               n2 = npgfb(jset)*(ncoset(lb_max(jset)) - ncoset(lb_min(jset) - 1))
               sgfb = first_sgfb(1, jset)

               ! calculate integrals and derivatives
               CALL overlap_ab(la_max(iset), la_min(iset), npgfa(iset), rpgfa(:, iset), zeta(:, iset), &
                               lb_max(jset), lb_min(jset), npgfb(jset), rpgfb(:, jset), zetb(:, jset), &
                               rab, sab=oint(:, :, 1))
               ! Contraction
               CALL contraction(oint(:, :, 1), owork, ca=scon_a(:, sgfa:), na=n1, ma=nsgfa(iset), &
                                cb=scon_b(:, sgfb:), nb=n2, mb=nsgfb(jset), fscale=1.0_dp, trans=trans)
!$OMP CRITICAL(blockadd)
               CALL block_add("IN", owork, nsgfa(iset), nsgfb(jset), sint(1)%block, &
                              sgfa, sgfb, trans=trans)
!$OMP END CRITICAL(blockadd)

            END DO
         END DO

      END DO
      DEALLOCATE (oint, owork)
      DEALLOCATE (sint)
!$OMP DO
!$    DO lock_num = 1, nlock
!$       call omp_destroy_lock(locks(lock_num))
!$    END DO
!$OMP END DO

!$OMP SINGLE
!$    DEALLOCATE (locks)
!$OMP END SINGLE NOWAIT

!$OMP END PARALLEL

      CALL dbcsr_finalize(matrix_s(1)%matrix)
      CALL dbcsr_filter(matrix_s(1)%matrix, dft_control%qs_control%eps_filter_matrix)

      CALL timestop(handle)

   END SUBROUTINE build_overlap_matrix_simple

! **************************************************************************************************
!> \brief   Calculation of the force contribution from an overlap matrix
!>          over Cartesian Gaussian functions.
!> \param   ks_env the QS environment
!> \param   force holds the calcuated force Tr(P dS/dR)
!> \param   basis_type_a basis set to be used for bra in <a|b>
!> \param   basis_type_b basis set to be used for ket in <a|b>
!> \param   sab_nl pair list (must be consistent with basis sets!)
!> \param   matrix_p density matrix for force calculation
!> \date    11.03.2002
!> \par     History
!>          Enlarged functionality of this routine. Now overlap matrices based
!>          on different basis sets can be calculated, taking into account also
!>          mixed overlaps NOTE: the pointer to the overlap matrix must now be
!>          put into its corresponding env outside of this routine
!>          [Manuel Guidon]
!>          Generalized for derivatives and force calculations [JHU]
!>          This special version is only for forces [07.07.2014, JGH]
!> \author  MK/JGH
!> \version 1.0
! **************************************************************************************************
   SUBROUTINE build_overlap_force(ks_env, force, basis_type_a, basis_type_b, &
                                  sab_nl, matrix_p)

      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      REAL(KIND=dp), DIMENSION(:, :), INTENT(INOUT)      :: force
      CHARACTER(LEN=*), INTENT(IN)                       :: basis_type_a, basis_type_b
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: sab_nl
      TYPE(dbcsr_type), POINTER                          :: matrix_p

      CHARACTER(len=*), PARAMETER :: routineN = 'build_overlap_force'

      INTEGER                                            :: handle, iatom, icol, ikind, irow, iset, &
                                                            jatom, jkind, jset, ldsab, n1, n2, &
                                                            natom, ncoa, ncob, nder, nkind, nseta, &
                                                            nsetb, sgfa, sgfb, slot
      INTEGER, DIMENSION(:), POINTER                     :: la_max, la_min, lb_max, lb_min, npgfa, &
                                                            npgfb, nsgfa, nsgfb
      INTEGER, DIMENSION(:, :), POINTER                  :: first_sgfa, first_sgfb
      LOGICAL                                            :: do_symmetric, found, trans, use_virial
      REAL(KIND=dp)                                      :: dab, f0, ff, rab2
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :)        :: pab, sab
      REAL(KIND=dp), ALLOCATABLE, DIMENSION(:, :, :)     :: drab
      REAL(KIND=dp), DIMENSION(3)                        :: force_a, rab
      REAL(KIND=dp), DIMENSION(3, 3)                     :: virial_thread
      REAL(KIND=dp), DIMENSION(:), POINTER               :: set_radius_a, set_radius_b
      REAL(KIND=dp), DIMENSION(3, SIZE(force, 2))        :: force_thread
      REAL(KIND=dp), DIMENSION(:, :), POINTER            :: p_block, rpgfa, rpgfb, scon_a, scon_b, &
                                                            zeta, zetb
      TYPE(gto_basis_set_p_type), DIMENSION(:), POINTER  :: basis_set_list_a, basis_set_list_b
      TYPE(gto_basis_set_type), POINTER                  :: basis_set_a, basis_set_b
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set
      TYPE(virial_type), POINTER                         :: virial

      CALL timeset(routineN, handle)

      NULLIFY (qs_kind_set)
      CALL get_ks_env(ks_env=ks_env, qs_kind_set=qs_kind_set)

      nkind = SIZE(qs_kind_set)
      nder = 1

      ! check for symmetry
      CPASSERT(SIZE(sab_nl) > 0)
      CALL get_neighbor_list_set_p(neighbor_list_sets=sab_nl, symmetric=do_symmetric)

      CALL get_ks_env(ks_env=ks_env, virial=virial)
      use_virial = virial%pv_availability .AND. (.NOT. virial%pv_numer)
      virial_thread = 0.0_dp

      ! set up basis sets
      ALLOCATE (basis_set_list_a(nkind), basis_set_list_b(nkind))
      CALL basis_set_list_setup(basis_set_list_a, basis_type_a, qs_kind_set)
      CALL basis_set_list_setup(basis_set_list_b, basis_type_b, qs_kind_set)
      ldsab = get_memory_usage(qs_kind_set, basis_type_a, basis_type_b)

      natom = SIZE(force, 2)
      force_thread = 0.0_dp

!$OMP PARALLEL DEFAULT(NONE) &
!$OMP SHARED (ldsab, do_symmetric, ncoset, nder, use_virial, force, virial, ndod, sab_nl, &
!$OMP         matrix_p, basis_set_list_a, basis_set_list_b)  &
!$OMP PRIVATE (ikind, jkind, iatom, jatom, rab, &
!$OMP          basis_set_a, basis_set_b, sab, pab, drab, &
!$OMP          first_sgfa, la_max, la_min, npgfa, nsgfa, nseta, rpgfa, set_radius_a, ncoa, ncob, force_a, &
!$OMP          zeta, first_sgfb, lb_max, lb_min, npgfb, nsetb, rpgfb, set_radius_b, nsgfb, p_block, dab, &
!$OMP          zetb, scon_a, scon_b, irow, icol, f0, ff, found, trans, rab2, n1, n2, sgfa, sgfb, iset, jset, &
!$OMP          slot) &
!$OMP REDUCTION (+ : virial_thread, force_thread )

      ! *** Allocate work storage ***
      ALLOCATE (sab(ldsab, ldsab), pab(ldsab, ldsab))
      ALLOCATE (drab(ldsab, ldsab, 3))

      ! Loop over neighbor list
!$OMP DO SCHEDULE(GUIDED)
      DO slot = 1, sab_nl(1)%nl_size
         ikind = sab_nl(1)%nlist_task(slot)%ikind
         jkind = sab_nl(1)%nlist_task(slot)%jkind
         iatom = sab_nl(1)%nlist_task(slot)%iatom
         jatom = sab_nl(1)%nlist_task(slot)%jatom
         rab(1:3) = sab_nl(1)%nlist_task(slot)%r(1:3)

         basis_set_a => basis_set_list_a(ikind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_a)) CYCLE
         basis_set_b => basis_set_list_b(jkind)%gto_basis_set
         IF (.NOT. ASSOCIATED(basis_set_b)) CYCLE
         ! basis ikind
         first_sgfa => basis_set_a%first_sgf
         la_max => basis_set_a%lmax
         la_min => basis_set_a%lmin
         npgfa => basis_set_a%npgf
         nseta = basis_set_a%nset
         nsgfa => basis_set_a%nsgf_set
         rpgfa => basis_set_a%pgf_radius
         set_radius_a => basis_set_a%set_radius
         scon_a => basis_set_a%scon
         zeta => basis_set_a%zet
         ! basis jkind
         first_sgfb => basis_set_b%first_sgf
         lb_max => basis_set_b%lmax
         lb_min => basis_set_b%lmin
         npgfb => basis_set_b%npgf
         nsetb = basis_set_b%nset
         nsgfb => basis_set_b%nsgf_set
         rpgfb => basis_set_b%pgf_radius
         set_radius_b => basis_set_b%set_radius
         scon_b => basis_set_b%scon
         zetb => basis_set_b%zet

         IF (do_symmetric) THEN
            IF (iatom <= jatom) THEN
               irow = iatom
               icol = jatom
            ELSE
               irow = jatom
               icol = iatom
            END IF
            f0 = 2.0_dp
            IF (iatom == jatom) f0 = 1.0_dp
            ff = 2.0_dp
         ELSE
            irow = iatom
            icol = jatom
            f0 = 1.0_dp
            ff = 1.0_dp
         END IF
         NULLIFY (p_block)
         CALL dbcsr_get_block_p(matrix=matrix_p, row=irow, col=icol, &
                                block=p_block, found=found)

         trans = do_symmetric .AND. (iatom > jatom)

         rab2 = rab(1)*rab(1) + rab(2)*rab(2) + rab(3)*rab(3)
         dab = SQRT(rab2)

         DO iset = 1, nseta

            ncoa = npgfa(iset)*ncoset(la_max(iset))
            n1 = npgfa(iset)*(ncoset(la_max(iset)) - ncoset(la_min(iset) - 1))
            sgfa = first_sgfa(1, iset)

            DO jset = 1, nsetb

               IF (set_radius_a(iset) + set_radius_b(jset) < dab) CYCLE

               ncob = npgfb(jset)*ncoset(lb_max(jset))
               n2 = npgfb(jset)*(ncoset(lb_max(jset)) - ncoset(lb_min(jset) - 1))
               sgfb = first_sgfb(1, jset)

               IF (ASSOCIATED(p_block) .AND. ((iatom /= jatom) .OR. use_virial)) THEN
                  ! Decontract P matrix block
                  sab = 0.0_dp
                  CALL block_add("OUT", sab, nsgfa(iset), nsgfb(jset), p_block, sgfa, sgfb, trans=trans)
                  CALL decontraction(sab, pab, scon_a(:, sgfa:), n1, nsgfa(iset), scon_b(:, sgfb:), n2, &
                                     nsgfb(jset), trans=trans)
                  ! calculate integrals and derivatives
                  CALL overlap_ab(la_max(iset), la_min(iset), npgfa(iset), rpgfa(:, iset), zeta(:, iset), &
                                  lb_max(jset), lb_min(jset), npgfb(jset), rpgfb(:, jset), zetb(:, jset), &
                                  rab, dab=drab)
                  CALL force_trace(force_a, drab, pab, n1, n2, 3)
                  force_thread(1:3, iatom) = force_thread(1:3, iatom) - ff*force_a(1:3)
                  force_thread(1:3, jatom) = force_thread(1:3, jatom) + ff*force_a(1:3)
                  IF (use_virial) THEN
                     CALL virial_pair_force(virial_thread, -f0, force_a, rab)
                  END IF
               END IF

            END DO
         END DO

      END DO
      DEALLOCATE (sab, pab, drab)
!$OMP END PARALLEL

!$OMP PARALLEL DEFAULT(NONE) &
!$OMP SHARED(force,force_thread,natom)
!$OMP WORKSHARE
      force(1:3, 1:natom) = force(1:3, 1:natom) + force_thread(1:3, 1:natom)
!$OMP END WORKSHARE
!$OMP END PARALLEL
      IF (use_virial) THEN
         virial%pv_virial = virial%pv_virial + virial_thread
         virial%pv_overlap = virial%pv_overlap + virial_thread
      END IF

      DEALLOCATE (basis_set_list_a, basis_set_list_b)

      CALL timestop(handle)

   END SUBROUTINE build_overlap_force

! **************************************************************************************************
!> \brief Setup the structure of a sparse matrix based on the overlap
!>        neighbor list
!> \param ks_env         The QS environment
!> \param matrix_s       Matrices to be constructed
!> \param matrix_name    Matrix base name
!> \param basis_set_list_a Basis set used for <a|
!> \param basis_set_list_b Basis set used for |b>
!> \param sab_nl         Overlap neighbor list
!> \param symmetric      Is symmetry used in the neighbor list?
! **************************************************************************************************
   SUBROUTINE create_sab_matrix_1d(ks_env, matrix_s, matrix_name, &
                                   basis_set_list_a, basis_set_list_b, sab_nl, symmetric)

      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      TYPE(dbcsr_p_type), DIMENSION(:), POINTER          :: matrix_s
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL             :: matrix_name
      TYPE(gto_basis_set_p_type), DIMENSION(:), POINTER  :: basis_set_list_a, basis_set_list_b
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: sab_nl
      LOGICAL, INTENT(IN)                                :: symmetric

      CHARACTER(LEN=12)                                  :: cgfsym
      CHARACTER(LEN=32)                                  :: symmetry_string
      CHARACTER(LEN=default_string_length)               :: mname, name
      INTEGER                                            :: i, maxs, natom
      INTEGER, DIMENSION(:), POINTER                     :: col_blk_sizes, row_blk_sizes
      TYPE(dbcsr_distribution_type), POINTER             :: dbcsr_dist
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set

      CALL get_ks_env(ks_env=ks_env, particle_set=particle_set, &
                      qs_kind_set=qs_kind_set, dbcsr_dist=dbcsr_dist)

      natom = SIZE(particle_set)

      IF (PRESENT(matrix_name)) THEN
         mname = matrix_name
      ELSE
         mname = "DUMMY"
      END IF

      maxs = SIZE(matrix_s)

      ALLOCATE (row_blk_sizes(natom), col_blk_sizes(natom))

      CALL get_particle_set(particle_set, qs_kind_set, nsgf=row_blk_sizes, &
                            basis=basis_set_list_a)
      CALL get_particle_set(particle_set, qs_kind_set, nsgf=col_blk_sizes, &
                            basis=basis_set_list_b)

      ! prepare for allocation
      IF (symmetric) THEN
         symmetry_string = dbcsr_type_symmetric
      ELSE
         symmetry_string = dbcsr_type_no_symmetry
      END IF

      DO i = 1, maxs
         IF (symmetric) THEN
            IF (ndod(i) == 1) THEN
               ! odd derivatives are anti-symmetric
               symmetry_string = dbcsr_type_antisymmetric
            ELSE
               symmetry_string = dbcsr_type_symmetric
            END IF
         ELSE
            symmetry_string = dbcsr_type_no_symmetry
         END IF
         cgfsym = cgf_symbol(1, indco(1:3, i))
         IF (i == 1) THEN
            name = mname
         ELSE
            name = TRIM(cgfsym(4:))//" DERIVATIVE OF THE "//TRIM(mname)// &
                   " W.R.T. THE NUCLEAR COORDINATES"
         END IF
         CALL compress(name)
         CALL uppercase(name)
         ALLOCATE (matrix_s(i)%matrix)
         CALL dbcsr_create(matrix=matrix_s(i)%matrix, &
                           name=TRIM(name), &
                           dist=dbcsr_dist, matrix_type=symmetry_string, &
                           row_blk_size=row_blk_sizes, col_blk_size=col_blk_sizes, &
                           nze=0)
         CALL cp_dbcsr_alloc_block_from_nbl(matrix_s(i)%matrix, sab_nl)
      END DO

      DEALLOCATE (row_blk_sizes, col_blk_sizes)

   END SUBROUTINE create_sab_matrix_1d

! **************************************************************************************************
!> \brief Setup the structure of a sparse matrix based on the overlap
!>        neighbor list, 2d version
!> \param ks_env         The QS environment
!> \param matrix_s       Matrices to be constructed
!> \param matrix_name    Matrix base name
!> \param basis_set_list_a Basis set used for <a|
!> \param basis_set_list_b Basis set used for |b>
!> \param sab_nl         Overlap neighbor list
!> \param symmetric      Is symmetry used in the neighbor list?
! **************************************************************************************************
   SUBROUTINE create_sab_matrix_2d(ks_env, matrix_s, matrix_name, &
                                   basis_set_list_a, basis_set_list_b, sab_nl, symmetric)

      TYPE(qs_ks_env_type), POINTER                      :: ks_env
      TYPE(dbcsr_p_type), DIMENSION(:, :), POINTER       :: matrix_s
      CHARACTER(LEN=*), INTENT(IN), OPTIONAL             :: matrix_name
      TYPE(gto_basis_set_p_type), DIMENSION(:), POINTER  :: basis_set_list_a, basis_set_list_b
      TYPE(neighbor_list_set_p_type), DIMENSION(:), &
         POINTER                                         :: sab_nl
      LOGICAL, INTENT(IN)                                :: symmetric

      CHARACTER(LEN=12)                                  :: cgfsym
      CHARACTER(LEN=32)                                  :: symmetry_string
      CHARACTER(LEN=default_string_length)               :: mname, name
      INTEGER                                            :: i1, i2, natom
      INTEGER, DIMENSION(:), POINTER                     :: col_blk_sizes, row_blk_sizes
      TYPE(dbcsr_distribution_type), POINTER             :: dbcsr_dist
      TYPE(particle_type), DIMENSION(:), POINTER         :: particle_set
      TYPE(qs_kind_type), DIMENSION(:), POINTER          :: qs_kind_set

      CALL get_ks_env(ks_env=ks_env, particle_set=particle_set, &
                      qs_kind_set=qs_kind_set, dbcsr_dist=dbcsr_dist)

      natom = SIZE(particle_set)

      IF (PRESENT(matrix_name)) THEN
         mname = matrix_name
      ELSE
         mname = "DUMMY"
      END IF

      ALLOCATE (row_blk_sizes(natom), col_blk_sizes(natom))

      CALL get_particle_set(particle_set, qs_kind_set, nsgf=row_blk_sizes, &
                            basis=basis_set_list_a)
      CALL get_particle_set(particle_set, qs_kind_set, nsgf=col_blk_sizes, &
                            basis=basis_set_list_b) 
      ! prepare for allocation
      IF (symmetric) THEN
         symmetry_string = dbcsr_type_symmetric
      ELSE
         symmetry_string = dbcsr_type_no_symmetry
      END IF
 
      DO i2 = 1, SIZE(matrix_s, 2)
         DO i1 = 1, SIZE(matrix_s, 1)
            IF (symmetric) THEN
               IF (ndod(i1) == 1) THEN
                  ! odd derivatives are anti-symmetric
                  symmetry_string = dbcsr_type_antisymmetric
               ELSE
                  symmetry_string = dbcsr_type_symmetric
               END IF
            ELSE
               symmetry_string = dbcsr_type_no_symmetry
            END IF
            cgfsym = cgf_symbol(1, indco(1:3, i1))
            IF (i1 == 1) THEN
               name = mname
            ELSE
               name = TRIM(cgfsym(4:))//" DERIVATIVE OF THE "//TRIM(mname)// &
                     " W.R.T. THE NUCLEAR COORDINATES"
            END IF
            CALL compress(name)
            CALL uppercase(name)
            ALLOCATE (matrix_s(i1, i2)%matrix)
            CALL dbcsr_create(matrix=matrix_s(i1, i2)%matrix, &
                              name=TRIM(name), &
                              dist=dbcsr_dist, matrix_type=symmetry_string, &
                              row_blk_size=row_blk_sizes, col_blk_size=col_blk_sizes, &
                              nze=0)
            CALL cp_dbcsr_alloc_block_from_nbl(matrix_s(i1, i2)%matrix, sab_nl)
         END DO
      END DO

      DEALLOCATE (row_blk_sizes, col_blk_sizes)

   END SUBROUTINE create_sab_matrix_2d

END MODULE qs_overlap

