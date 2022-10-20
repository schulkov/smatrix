MODULE mathlib
USE kinds,         ONLY: dp
USE mathconstants, ONLY: fac
IMPLICIT NONE
PRIVATE
   PUBLIC :: binomial

CONTAINS

! **************************************************************************************************
!> \brief   The binomial coefficient n over k for 0 <= k <= n is calculated,
!>            otherwise zero is returned.
! **************************************************************************************************
   PURE FUNCTION binomial(n, k) RESULT(n_over_k)
      INTEGER, INTENT(in)                                :: n, k
      REAL(kind=dp)                                      :: n_over_k

      IF ((k >= 0) .AND. (k <= n)) THEN
         n_over_k = fac(n)/(fac(n-k)*fac(k))
      ELSE
         n_over_k = 0.0_dp
      END IF

   END FUNCTION binomial
END MODULE mathlib
