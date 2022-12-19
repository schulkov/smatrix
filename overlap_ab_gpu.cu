#include <stdlib.h>
#include <stdio.h>

#define GPU_ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__host__ __device__ int ncoset( int l_max ){
  int nco = 0;
  for( int l=0; l <= l_max; l++ ){
    nco += (l+1)*(l+2)/2;
  }
  return nco;
}

__host__ __device__ int coset( int lx, int ly, int lz ){
  int l = lx + ly + lz;
  int co = 1 + (l - lx)*(l - lx + 1)/2 + lz;
  return ncoset(l - 1) + co - 1 ;
}

__host__ __device__ inline unsigned int get_nco(int l){
   unsigned int nco = 0;
   if (l >= 0) nco = (l+1)*(l+2)/2;
   return nco;
}

__host__ __device__ void os_rr_ovlp( double rap[3], int la_max, double rbp[3], int lb_max, double zet, int ldrr, double* rr ){
   // uses the Obara-Saika Recurrence Relations to compute integrals of the type 
   // (x-Ax)^La (x-Bx)^Lb exp(-zeta(r-A)**2) exp(-zetb(r-B)**2)
   // usually, these integrals are then used to compute the (r-A) integrals
   // and their derivatives using separation of variables
   double g = 0.5/zet;
   // s integrals
   rr[0] = 1.0;
   rr[1] = 1.0;
   rr[2] = 1.0;
   //!
   //! recursion along la for lb=0
   //!
   if (la_max > 0) {
      // <p|s> = (p-a) <s|s> . Ref OS table VI line 2
      rr[ldrr*3+0] = rap[0];
      rr[ldrr*3+1] = rap[1];
      rr[ldrr*3+2] = rap[2];
      //!
      for( int la = 1; la < la_max ; la++ ){ // DO la = 1, la_max - 1
         int lap1 = la + 1;
         int lam1 = la - 1;
         // <a+1|s> = 1/2z N(a) <a-1|s> + (p-a) <a|b> . Ref OS eq A2, but in reverse order (?)
         rr[lap1*ldrr*3+0] = double(la)*g*rr[lam1*ldrr*3+0] + rap[0]*rr[la*ldrr*3+0];
         rr[lap1*ldrr*3+1] = double(la)*g*rr[lam1*ldrr*3+1] + rap[1]*rr[la*ldrr*3+1];
         rr[lap1*ldrr*3+2] = double(la)*g*rr[lam1*ldrr*3+2] + rap[2]*rr[la*ldrr*3+2];
      }
   }
   //!
   //! recursion along lb for all la
   //!
   if (lb_max > 0) {
      // <s|p> = (p-b) <s|s>
      rr[1*3+0] = rbp[0];
      rr[1*3+1] = rbp[1];
      rr[1*3+2] = rbp[2];
      //!
      for( int la=1 ; la <= la_max ; la++ ){ // DO la = 1, la_max
         int lam1 = la - 1;
         // <a|p> = <a|s+1> = 1/2z Na <a-1|s> + (p-b) <a|s> . Ref OS eq A2 with b<=>a
         rr[(la*ldrr+1)*3+0] = double(la)*g*rr[lam1*ldrr*3+0] + rbp[0]*rr[la*ldrr*3+0];
         rr[(la*ldrr+1)*3+1] = double(la)*g*rr[lam1*ldrr*3+1] + rbp[1]*rr[la*ldrr*3+1];
         rr[(la*ldrr+1)*3+2] = double(la)*g*rr[lam1*ldrr*3+2] + rbp[2]*rr[la*ldrr*3+2];
      }
      //!
      for( int lb=1 ; lb < lb_max; lb++ ){ // DO lb = 1, lb_max - 1
         int lbp1 = lb + 1;
         int lbm1 = lb - 1;
         // <s|b+1> = 1/2z Nb <s|p-1> + (p-b) <s|p>. Ref OS eq A2 with b<=>a
         rr[lbp1*3+0] = double(lb)*g*rr[lbm1*3+0] + rbp[0]*rr[lb*3+0];
         rr[lbp1*3+1] = double(lb)*g*rr[lbm1*3+1] + rbp[1]*rr[lb*3+1];
         rr[lbp1*3+2] = double(lb)*g*rr[lbm1*3+2] + rbp[2]*rr[lb*3+2];
         for( int la=1; la <= la_max; la++ ){ // DO la = 1, la_max
            int lam1 = la - 1;
            // <a|b+1> = 1/2z Na <a-1|b> + 1/2z Nb <a|b-1> + (p-b) <a|b>
            rr[(la*ldrr+lbp1)*3+0] = g*(double(la)*rr[(lam1*ldrr+lb)*3+0] + double(lb)*rr[(la*ldrr+lbm1)*3+0]) + rbp[0]*rr[(la*ldrr+lb)*3+0];
            rr[(la*ldrr+lbp1)*3+1] = g*(double(la)*rr[(lam1*ldrr+lb)*3+1] + double(lb)*rr[(la*ldrr+lbm1)*3+1]) + rbp[1]*rr[(la*ldrr+lb)*3+1];
            rr[(la*ldrr+lbp1)*3+2] = g*(double(la)*rr[(lam1*ldrr+lb)*3+2] + double(lb)*rr[(la*ldrr+lbm1)*3+2]) + rbp[2]*rr[(la*ldrr+lb)*3+2];
         }
      }
   }
}

__host__ __device__ void overlap_ab_zeta( 
      int la_max, int la_min, int ipgfa, double rpgfa, double zeta, 
      int lb_max, int lb_min, int jpgfb, double rpgfb, double zetb,
      double* rab, double* sab, double* dab, double* ddab, int lds, double* rr, int ldrr )
{
   // computes the na*nb integrals int dr (r-A)^La (r-B)^Lb exp(-zeta(r-A)**2) exp(-zetb(r-B)**2)
   // for each combination of lax+lay+laz=[La_min:La_max] and lbx+lby+lbz=[Lb_min:lb_max]
   // saves them in the sub-matrix sab[ na*ipgfa ][ nb*jpgfb ]
   int ofa = ncoset(la_min - 1);
   int ofb = ncoset(lb_min - 1);
   int na = ncoset(la_max) - ofa;
   int nb = ncoset(lb_max) - ofb;
   int lma = la_max;
   int lmb = lb_max;
 
   int ma = na * ipgfa;
   int mb = nb * jpgfb;
//   lds = int nb = npgfb*(ncoset(lb_max)-ncoset(lb_min-1));

   double rab2 = rab[0]*rab[0] + rab[1]*rab[1] + rab[2]*rab[2];
   double tab = sqrt(rab2);
   if ( (rpgfa + rpgfb) < tab ){
      for( int i=0; i<na; i++ ){
      for( int j=0; j<nb; j++ ){
         sab[ (ma+i)*lds+ mb+j ] = 0.0;
      }}
      return;
   }
   
   //! Calculate some prefactors
   double a = zeta;
   double b = zetb;
   double zet = a + b;
   double xhi = a*b/zet;
   double rap[3], rbp[3];
   rap[0] = b*rab[0]/zet;
   rap[1] = b*rab[1]/zet;
   rap[2] = b*rab[2]/zet;
   rbp[0] = -a*rab[0]/zet;
   rbp[1] = -a*rab[1]/zet;
   rbp[2] = -a*rab[2]/zet;
   //! [s|s] integral
   double pi_zet = M_PI/zet;
   double f0 = sqrt(pi_zet*pi_zet*pi_zet)*exp(-xhi*rab2);

   //! Calculate the recurrence relation
   os_rr_ovlp(rap, lma, rbp, lmb, zet, ldrr, rr);
   // la,ax,ay,az could be in the for expression to limit their scope, but it would be even (more) unreadable
   int la = la_min;
   int ax = la_min;
   int ay = 0;
   int az = 0;
   for( int coa = 0; coa < na; coa++ ){
      int ia = ma + coa ;
      int lb = lb_min;
      int bx = lb_min;
      int by = 0;
      int bz = 0;
      // 
      const double* const rr_ax = &rr[(ax*ldrr)*3];
      const double* const rr_ay = &rr[(ay*ldrr)*3];
      const double* const rr_az = &rr[(az*ldrr)*3];
      for( int cob = 0; cob < nb; cob++ ){
         int ib = mb + cob ;
         // contigous in access to sab wrt ib. Very not cont. when accessing the smaller rr[] arrays
         sab[ia*lds+ ib] = f0*rr_ax[bx*3+0]*rr_ay[by*3+1]*rr_az[bz*3+2];
         // next bx,by,bz,lb quartet
         if (bz == lb) {
            // if bz is equal to l it means we have computed all integrals for this l
            // so we move to the next l value and reset bx,by and bz
            lb += 1; bx = lb; bz = 0; // by = 0 is done automatically at the end
         } else {
           if (by == 0) {
               // if by == 0 it means we have exausted the [y,z] subspace and move to the next x
               bz = 0; bx -= 1;
           } else {
               // explore the current [y,z]=l-lx subspace
               bz += 1;
           }
         }
         // why bother incrementing and decrementing ly
         by = lb - bx - bz;
      }
      // next ax,ay,az,la quartet
      if (az == la) {
         la += 1; ax = la; az = 0;
      } else {
        if (ay == 0) {
            az = 0; ax -= 1;
        } else {
            az += 1;
        }
      }
      ay = la - ax - az;
   }
}

__global__ void overlap_ab_gpu_kernel( 
      int la_max, int la_min, int npgfa, double* rpgfa, double* zeta,
      int lb_max, int lb_min, int npgfb, double* rpgfb, double* zetb,
      double* rab, double* sab, double* dab, double* ddab, int lds, double* rr, int ldrr ){
   // if the thread is in the valid subset, assign it its zet coefficents and its workspace on rr
   int ipgfa = threadIdx.x;
   int jpgfb = threadIdx.y;
   if ( (ipgfa < npgfa) and (jpgfb < npgfb ) ){
      double* rr_subset = &(rr[ldrr*ldrr*3*(ipgfa*npgfb+jpgfb)]);
      overlap_ab_zeta(
         la_max, la_min, ipgfa, rpgfa[ipgfa], zeta[ipgfa],
         lb_max, lb_min, jpgfb, rpgfb[jpgfb], zetb[jpgfb],
         rab, sab, dab, ddab, lds, rr_subset, ldrr );
   }
}




extern "C" {


void overlap_ab_gpu(
      int la_max, int la_min, int npgfa, double* rpgfa, double* zeta,
      int lb_max, int lb_min, int npgfb, double* rpgfb, double* zetb,
      double* rab, double* sab, int lds )
{
   double *rr_dev = NULL;
   double *sab_dev = NULL;
   double *zeta_dev = NULL;
   double *zetb_dev = NULL;
   double *rpgfa_dev = NULL;
   double *rpgfb_dev = NULL;
   double *rab_dev = NULL;
   int lma = la_max;
   int lmb = lb_max;
   int ldrr = max(lma, lmb) + 1;
   int n1 = npgfa*(ncoset(la_max)-ncoset(la_min-1));
   int n2 = npgfb*(ncoset(lb_max)-ncoset(lb_min-1));

   GPU_ERROR_CHECK(cudaMalloc( (void**) &rr_dev, npgfa*npgfb*ldrr*ldrr*3*sizeof(double) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &sab_dev, n1*n2*sizeof(double) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &zeta_dev, npgfa*sizeof(double) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &zetb_dev, npgfb*sizeof(double) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &rpgfa_dev, npgfa*sizeof(double) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &rpgfb_dev, npgfb*sizeof(double) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &rab_dev, 3*sizeof(double) ));

   GPU_ERROR_CHECK(cudaMemset(   sab_dev, 0, n1*n2*sizeof(double)));  

   GPU_ERROR_CHECK(cudaMemcpy(  zeta_dev,  zeta, npgfa*sizeof(double), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemcpy(  zetb_dev,  zetb, npgfb*sizeof(double), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemcpy( rpgfa_dev, rpgfa, npgfa*sizeof(double), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemcpy( rpgfb_dev, rpgfb, npgfb*sizeof(double), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemcpy(   rab_dev,   rab, 3*sizeof(double), cudaMemcpyHostToDevice ));

   dim3 npgf_ab(npgfa, npgfb);
   overlap_ab_gpu_kernel<<<1, npgf_ab >>>(
         la_max, la_min, npgfa, rpgfa_dev, zeta_dev,
         lb_max, lb_min, npgfb, rpgfb_dev, zetb_dev,
         rab_dev, sab_dev, NULL, NULL, n2, rr_dev, ldrr );
   GPU_ERROR_CHECK(cudaGetLastError() );

   GPU_ERROR_CHECK(cudaMemcpy( sab, sab_dev, n1*n2*sizeof(double), cudaMemcpyDeviceToHost ));

   GPU_ERROR_CHECK(cudaFree(zeta_dev));
   GPU_ERROR_CHECK(cudaFree(zetb_dev));
   GPU_ERROR_CHECK(cudaFree(rpgfa_dev));
   GPU_ERROR_CHECK(cudaFree(rpgfb_dev));
   GPU_ERROR_CHECK(cudaFree(rab_dev));
   GPU_ERROR_CHECK(cudaFree(rr_dev));
   GPU_ERROR_CHECK(cudaFree(sab_dev));
}

} // end of extern C






























