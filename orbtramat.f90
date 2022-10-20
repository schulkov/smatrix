MODULE orbtramat
USE kinds, ONLY: dp
USE mathconstants, ONLY: fac
USE mathlib, ONLY: binomial
IMPLICIT NONE
PRIVATE

   PUBLIC :: get_nco, get_co, get_ncoset, get_coset
   PUBLIC :: get_nso, get_so,  get_nsoset, get_soset
   PUBLIC :: get_c2s, get_s2c

CONTAINS

   ! number of cartesian orbitals
   ELEMENTAL FUNCTION get_nco(l) RESULT(nco)
      INTEGER, INTENT(in)             :: l
      INTEGER                         :: nco

      IF (l >= 0) THEN
         nco = (l+1)*(l+2)/2
      ELSE
         nco = 0
      END IF
   END FUNCTION get_nco

   ! index of cartesian orbital derived from a primitive GTO with the angular momentum l
   PURE FUNCTION get_co(lx, ly, lz) RESULT(co)
      INTEGER, INTENT(in)             :: lx, ly, lz
      INTEGER                         :: co

      INTEGER :: l

      l = lx+ly+lz
      co = 1+(l-lx)*(l-lx+1)/2+lz
   END FUNCTION get_co

   ELEMENTAL FUNCTION get_ncoset(l) RESULT(ncoset)
      INTEGER, INTENT(in)             :: l
      INTEGER                         :: ncoset

      INTEGER :: ll

      ncoset = 0
      DO ll = 0,l
         ncoset = ncoset + get_nco(ll)
      END DO
   END FUNCTION get_ncoset

   PURE FUNCTION get_coset(lx, ly, lz) RESULT(coset)
      INTEGER, INTENT(in)             :: lx, ly, lz
      INTEGER                         :: coset

      INTEGER :: l

      l = lx+ly+lz
      coset = get_ncoset(l-1)+get_co(lx, ly, lz)
   END FUNCTION get_coset

   ! number of spherical orbitals
   ELEMENTAL FUNCTION get_nso(l) RESULT(nso)
      INTEGER, INTENT(in)             :: l
      INTEGER                         :: nso

      nso = 2*l+1
   END FUNCTION get_nso

   PURE FUNCTION get_so(l, m) RESULT(so)
      INTEGER, INTENT(in)             :: l, m
      INTEGER                         :: so

      so = get_nso(l)-(l-m)
   END FUNCTION get_so

   ELEMENTAL FUNCTION get_nsoset(l) RESULT(nsoset)
      INTEGER, INTENT(in)             :: l
      INTEGER                         :: nsoset

      INTEGER :: ll

      nsoset = 0
      DO ll = 0,l
         nsoset = nsoset + get_nso(ll)
      END DO
   END FUNCTION get_nsoset

   PURE FUNCTION get_soset(l, m) RESULT(soset)
      INTEGER, INTENT(in)             :: l, m
      INTEGER                         :: soset

      soset = get_nsoset(l-1)+get_so(l, m)
   END FUNCTION get_soset

   PURE FUNCTION get_c2s(l) RESULT(c2s)
      INTEGER, INTENT(in)                                       :: l
      REAL(kind=dp), DIMENSION(get_nso(l), get_nco(l))          :: c2s

      INTEGER :: expo, i, ic, is, j, k, lx, ly, lz, m, ma
      REAL(kind=dp) :: s, s1, s2

      c2s = 0.0_dp

!     *** Build the orbital transformation matrix for the     ***
!     *** transformation from Cartesian to spherical orbitals ***
!     *** (c2s, formula 15)                                   ***

      DO lx = 0, l
         DO ly = 0, l-lx
            lz = l-lx-ly
            ic = get_co(lx, ly, lz)
            DO m = -l, l
               is = l+m+1
               ma = ABS(m)
               j = lx+ly-ma
               IF ((j >= 0) .AND. (MODULO(j, 2) == 0)) THEN
                  j = j/2
                  s1 = 0.0_dp
                  DO i = 0, (l-ma)/2
                     s2 = 0.0_dp
                     DO k = 0, j
                        IF (((m < 0) .AND. (MODULO(ABS(ma-lx), 2) == 1)) .OR. &
                           ((m > 0) .AND. (MODULO(ABS(ma-lx), 2) == 0))) THEN
                           expo = (ma-lx+2*k)/2
                           s = (-1.0_dp)**expo*SQRT(2.0_dp)
                        ELSE IF ((m == 0) .AND. (MODULO(lx, 2) == 0)) THEN
                           expo = k-lx/2
                           s = (-1.0_dp)**expo
                        ELSE
                           s = 0.0_dp
                        END IF
                        s2 = s2+binomial(j, k)*binomial(ma, lx-2*k)*s
                     END DO
                     s1 = s1+binomial(l, i)*binomial(i, j)* &
                          (-1.0_dp)**i*fac(2*l-2*i)/fac(l-ma-2*i)*s2
                  END DO
                  c2s(is, ic) = &
                       SQRT((fac(2*lx)*fac(2*ly)*fac(2*lz)*fac(l)*fac(l-ma))/ &
                            (fac(lx)*fac(ly)*fac(lz)*fac(2*l)*fac(l+ma)))*s1/ &
                       (2.0_dp**l*fac(l))
               ELSE
                  c2s(is, ic) = 0.0_dp
               END IF
            END DO
         END DO
      END DO
   END FUNCTION get_c2s

   PURE FUNCTION get_s2c(l, c2s) RESULT(s2c)
      INTEGER, INTENT(in)                                       :: l
      REAL(kind=dp), DIMENSION(get_nso(l), get_nco(l)), INTENT(in) :: c2s
      REAL(kind=dp), DIMENSION(get_nso(l), get_nco(l))          :: s2c

      INTEGER :: ic1, ic2, is, lx, lx1, lx2, ly, ly1, ly2, lz, lz1, lz2, nso
      REAL(kind=dp) :: s, s1, s2

      s2c = 0.0_dp
      nso = get_nso(l)

!     *** Build the corresponding transformation matrix for the ***
!     *** transformation from spherical to Cartesian orbitals   ***
!     *** (s2c = s*TRANSPOSE(c2s), formulas 18 and 19)          ***

      DO lx1 = 0, l
         DO ly1 = 0, l-lx1
            lz1 = l-lx1-ly1
            ic1 = get_co(lx1, ly1, lz1)
            s1 = SQRT((fac(lx1)*fac(ly1)*fac(lz1))/ &
                      (fac(2*lx1)*fac(2*ly1)*fac(2*lz1)))
            DO lx2 = 0, l
               DO ly2 = 0, l-lx2
                  lz2 = l-lx2-ly2
                  ic2 = get_co(lx2, ly2, lz2)
                  lx = lx1+lx2
                  ly = ly1+ly2
                  lz = lz1+lz2
                  IF ((MODULO(lx, 2) == 0) .AND. &
                      (MODULO(ly, 2) == 0) .AND. &
                      (MODULO(lz, 2) == 0)) THEN
                     s2 = SQRT((fac(lx2)*fac(ly2)*fac(lz2))/ &
                               (fac(2*lx2)*fac(2*ly2)*fac(2*lz2)))
                     s = fac(lx)*fac(ly)*fac(lz)*s1*s2/ &
                         (fac(lx/2)*fac(ly/2)*fac(lz/2))
                     DO is = 1, nso
                        s2c(is, ic1) = s2c(is, ic1)+s*c2s(is, ic2)
                     END DO
                  END IF
               END DO
            END DO
         END DO
      END DO


   END FUNCTION get_s2c

END MODULE orbtramat
