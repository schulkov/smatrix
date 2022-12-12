/* exp(), sqrt() */
#include <math.h>
/* malloc(), free() */
#include <stdlib.h>
/* memset() */
#include <string.h>
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



/*
    few special cases derived from CP2K general subroutine
    https://github.com/cp2k/cp2k/blob/master/src/aobasis/ai_overlap.F
*/

__host__ __device__ void overlap_primitive_ss(double *sab, double zeta, double zetb, double rab_x, double rab_y, double rab_z)
{
   double dab = sqrt(rab_x*rab_x + rab_y*rab_y + rab_z*rab_z);

   //  *** Prefactors ***
   double zetp = ((double)1.0)/(zeta+zetb);
   double pi_zetp = M_PI * zetp;
   double f0 = pi_zetp*sqrt(pi_zetp);
   double f1 = zetb*zetp;

   *sab = f0*exp(-zeta*f1*dab*dab);
}

__host__ __device__ void overlap_primitive_sp(double *sab, double zeta, double zetb, double rab_x, double rab_y, double rab_z)
{
   double dab = sqrt(rab_x*rab_x + rab_y*rab_y + rab_z*rab_z);

   //  *** Prefactors ***
   double zetp = ((double)1.0)/(zeta+zetb);
   double pi_zetp = M_PI * zetp;
   double f0 = sqrt(pi_zetp*pi_zetp*pi_zetp);
   double f1 = zetb*zetp;
   double f1_m1 = f1 - (double)1.0;
   double rbp_x = f1_m1*rab_x;
   double rbp_y = f1_m1*rab_y;
   double rbp_z = f1_m1*rab_z;

   double s0 = f0*exp(-zeta*f1*dab*dab); // [s|s]
   sab[0] = rbp_x*s0; // [s|px]
   sab[1] = rbp_y*s0; // [s|py]
   sab[2] = rbp_z*s0; // [s|pz]
}

/* the same as overlap_primitive_sp(sab, zetb, zeta, rab_x, rab_y, rab_z) due to symmetry */
__host__ __device__ void overlap_primitive_ps(double *sab, double zeta, double zetb, double rab_x, double rab_y, double rab_z)
{
   double dab = sqrt(rab_x*rab_x + rab_y*rab_y + rab_z*rab_z);

   //  *** Prefactors ***
   double zetp = ((double)1.0)/(zeta+zetb);
   double pi_zetp = M_PI * zetp;
   double f0 = sqrt(pi_zetp*pi_zetp*pi_zetp);
   double f1 = zetb*zetp;
   double rap_x = f1*rab_x;
   double rap_y = f1*rab_y;
   double rap_z = f1*rab_z;

   double s0 = f0*exp(-zeta*f1*dab*dab); // [s|s]
   sab[0] = rap_x*s0; // [px|s]
   sab[1] = rap_y*s0; // [py|s]
   sab[2] = rap_z*s0; // [pz|s]
}

__host__ __device__ void overlap_primitive_pp(double *sab, double zeta, double zetb, double rab_x, double rab_y, double rab_z)
{
   double dab = sqrt(rab_x*rab_x + rab_y*rab_y + rab_z*rab_z);

   //  *** Prefactors ***
   double zetp = ((double)1.0)/(zeta+zetb);
   double pi_zetp = M_PI * zetp;
   double f0 = sqrt(pi_zetp*pi_zetp*pi_zetp);
   double f1 = zetb*zetp;
   double f2 = ((double)0.5)*zetp;
   double rap_x = f1*rab_x;
   double rap_y = f1*rab_y;
   double rap_z = f1*rab_z;
   double rbp_x = rap_x-rab_x;
   double rbp_y = rap_y-rab_y;
   double rbp_z = rap_z-rab_z;
   double s0, s1;

   s0 = f0*exp(-zeta*f1*dab*dab); // [s|s]

   s1 = rap_x*s0; // [px|s]
   sab[0] = rbp_x*s1+f2*s0; // [px|px]
   sab[1] = rbp_y*s1; // [px|py]
   sab[2] = rbp_z*s1; // [px|pz]

   s1 = rap_y*s0; // [py|s]
   sab[3] = rbp_x*s1; // [py|px]
   sab[4] = rbp_y*s1+f2*s0; // [py|py]
   sab[5] = rbp_z*s1; // [py|pz]

   s1 = rap_z*s0; // [pz|s]
   sab[6] = rbp_x*s1; // [pz|px]
   sab[7] = rbp_y*s1; // [pz|py]
   sab[8] = rbp_z*s1+f2*s0; // [pz|pz]
}


__global__ void overlap_ab_cgf_kernel(
    double* sab_dev, double* sab_pgf_dev, double* gcc_a_dev, double* gcc_b_dev, double* zet_a_dev, double* zet_b_dev,
    int la_set, int lb_set, unsigned int ncoa, unsigned int ncob, double rab_x, double rab_y, double rab_z )
{
   unsigned int ipgf_a = threadIdx.x ;
   unsigned int ipgf_b = threadIdx.y ;
   int npgf_a = blockDim.x;
   int npgf_b = blockDim.y;
   double gccSgcc_ab;
   /*
      For each pair of primitives ipgf_a,ipgf_b, fill S_ab_pgf with the correct polynomial[r,e-mr2]
      Once that is done, accumulate the S_ab = <c_a|S_ab_pgf|c_b> on the ncoa,ncob matrix
      With  S_ab_pgf = product between primitive gaussian functions
            S_ab     = product between contracted gaussian functions
            c_a      = contraction coefficients of a
   */
   if (la_set == 0 && lb_set == 0) {
      overlap_primitive_ss(&sab_pgf_dev[(ipgf_a*npgf_b+ipgf_b)*ncoa*ncob], zet_a_dev[ipgf_a], zet_b_dev[ipgf_b], rab_x, rab_y, rab_z);
   } else if (la_set == 0 && lb_set == 1) {
      overlap_primitive_sp(&sab_pgf_dev[(ipgf_a*npgf_b+ipgf_b)*ncoa*ncob], zet_a_dev[ipgf_a], zet_b_dev[ipgf_b], rab_x, rab_y, rab_z);
   } else if (la_set == 1 && lb_set == 0) {
      overlap_primitive_ps(&sab_pgf_dev[(ipgf_a*npgf_b+ipgf_b)*ncoa*ncob], zet_a_dev[ipgf_a], zet_b_dev[ipgf_b], rab_x, rab_y, rab_z);
   } else if (la_set == 1 && lb_set == 1) {
      overlap_primitive_pp(&sab_pgf_dev[(ipgf_a*npgf_b+ipgf_b)*ncoa*ncob], zet_a_dev[ipgf_a], zet_b_dev[ipgf_b], rab_x, rab_y, rab_z);
   }

   for (unsigned int icob = 0; icob < ncob; ++icob) {
      for (unsigned int icoa = 0; icoa < ncoa; ++icoa) {
         gccSgcc_ab = sab_pgf_dev[(ipgf_a*npgf_b+ipgf_b)*ncoa*ncob+icob*ncoa+icoa] * 
                         gcc_a_dev[icoa*npgf_a+ipgf_a] * gcc_b_dev[icob*npgf_b+ipgf_b];
         atomicAdd_block(&sab_dev[icob*ncoa+icoa], gccSgcc_ab);
      }
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
      rr[ldrr*3+0] = rbp[0];
      rr[ldrr*3+1] = rbp[1];
      rr[ldrr*3+2] = rbp[2];
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
            rr[(la*ldrr+lbp1)*3+0] = g*(double(la)*rr[(lam1*ldrr+lb)*3+0] + double(lb)*rr[(la*ldrr+lbm1)*3+0]) + rbp[0]*rr[(la*ldrr+lb)+0];
            rr[(la*ldrr+lbp1)*3+1] = g*(double(la)*rr[(lam1*ldrr+lb)*3+1] + double(lb)*rr[(la*ldrr+lbm1)*3+1]) + rbp[1]*rr[(la*ldrr+lb)+1];
            rr[(la*ldrr+lbp1)*3+2] = g*(double(la)*rr[(lam1*ldrr+lb)*3+2] + double(lb)*rr[(la*ldrr+lbm1)*3+2]) + rbp[2]*rr[(la*ldrr+lb)+2];
         }
      }
   }
}

__host__ __device__ void next_Lxyz( int* __restrict__ lx, int* __restrict__ ly, int* __restrict__ lz, int* __restrict__ l ){
   if (*lz == *l) {
      (*l)++; (*lx) = (*l); (*lz) = 0;
   } else {
      if (*ly == 0) {
         (*lz) = 0; (*lx)--;
      } else {
         (*lz)++;
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
   for( int coa = ncoset(la_min); coa <= ncoset(la_max); coa++ ){
      int ia = ma + coa;
      int lb = lb_min;
      int bx = lb_min;
      int by = 0;
      int bz = 0;
      // 
      const double* const rr_ax = &rr[(ax*ldrr)*3];
      const double* const rr_ay = &rr[(ay*ldrr)*3];
      const double* const rr_az = &rr[(az*ldrr)*3];
      for( int cob = ncoset(lb_min); cob <= ncoset(lb_max); cob++ ){
         int ib = mb + cob;
         // contigous in access to sab. Very not cont. when accessing the smaller rr[] arrays
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

/*
   for( int lb = lb_min; lb <= lb_max; lb++ ){ // DO lb = lb_min, lb_max
   for( int bx = 0; bx <= lb ; bx++ ){ // DO bx = 0, lb
   for( int by = 0; by <= lb-bx; lb++ ){ // DO by = 0, lb - bx
      int bz = lb - bx - by;
      int cob = coset(bx, by, bz) - ofb;
      int ib = mb + cob;
      for( int la = la_min ; la <= la_max ; la++ ){ // DO la = la_min, la_max
      for( int ax = 0 ; ax <= la ; ax++ ){ // DO ax = 0, la
      for( int ay = 0; ay <= la-ax; ay++ ){ // DO ay = 0, la - ax
         int az = la - ax - ay;
         int coa = coset(ax, ay, az) - ofa;
         int ia = ma + coa;
         //! integrals
         // uses: int dx dy dz (x-Ax)^lax (y-Ay)^lay (z-Az)^laz exp(-zeta(r-A)**2)) * [ Same with B ] =
         //       = int dx (x-Ax)^lax (x-Bx)^lbx exp( -zeta (x-Ax)^2 ) exp( -zetb (x-Bx)^2 ) * [ Same on y and z ]
//       sab[ia][ib]     = f0*rr[ax][bx][0]         *rr[ay][by][1]         *rr[az][bz][2];
         sab[ia*lds+ ib] = f0*rr[(ax*ldrr+bx)*3+0]*rr[(ay*ldrr+by)*3+1]*rr[(az*ldrr+bz)*3+2];
      }}}
   }}}
*/

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
      double* rab, double* sab, double* dab, double* ddab, int lds )
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

   GPU_ERROR_CHECK(cudaMalloc( (void**) &rr_dev, npgfa*npgfb*ldrr*ldrr*3*sizeof(double) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &sab_dev, lds*lds*sizeof(double) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &zeta_dev, npgfa*sizeof(double) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &zetb_dev, npgfb*sizeof(double) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &rpgfa_dev, npgfa*sizeof(double) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &rpgfb_dev, npgfb*sizeof(double) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &rab_dev, 3*sizeof(double) ));

   GPU_ERROR_CHECK(cudaMemset(   sab_dev, 0, lds*lds*sizeof(double)));  

   GPU_ERROR_CHECK(cudaMemcpy(  zeta_dev,  zeta, npgfa*sizeof(double), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemcpy(  zetb_dev,  zetb, npgfb*sizeof(double), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemcpy( rpgfa_dev, rpgfa, npgfa*sizeof(double), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemcpy( rpgfb_dev, rpgfb, npgfb*sizeof(double), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemcpy(   rab_dev,   rab, 3*sizeof(double), cudaMemcpyHostToDevice ));

   dim3 npgf_ab(npgfa, npgfb);
   overlap_ab_gpu_kernel<<<1, npgf_ab >>>(
         la_max, la_min, npgfa, rpgfa_dev, zeta_dev,
         lb_max, lb_min, npgfb, rpgfb_dev, zetb_dev,
         rab_dev, sab_dev, NULL, NULL, lds, rr_dev, ldrr );
   GPU_ERROR_CHECK(cudaGetLastError() );

   GPU_ERROR_CHECK(cudaMemcpy( sab, sab_dev, lds*lds*sizeof(double), cudaMemcpyDeviceToHost ));

   GPU_ERROR_CHECK(cudaFree(zeta_dev));
   GPU_ERROR_CHECK(cudaFree(zetb_dev));
   GPU_ERROR_CHECK(cudaFree(rpgfa_dev));
   GPU_ERROR_CHECK(cudaFree(rpgfb_dev));
   GPU_ERROR_CHECK(cudaFree(rab_dev));
   GPU_ERROR_CHECK(cudaFree(rr_dev));
   GPU_ERROR_CHECK(cudaFree(sab_dev));
}

/*
   overlap integral v1, unoptmized
   sab : overlap matrix element over contracted Gaussian functions
   la_set, lb_set : angular momenta
   npgf_a, npgf_b : number of primitive Gaussian functions in contracted sets
   zet_a(1:npgf_a), zet_b(1:npgf_b) : Gaussian exponents
         rab_dev, sab_dev, NULL, NULL, lds, rr_dev, ldrr );
   GPU_ERROR_CHECK(cudaGetLastError() );

   GPU_ERROR_CHECK(cudaMemcpy( sab, sab_dev, lds*lds*sizeof(double), cudaMemcpyDeviceToHost ));

   GPU_ERROR_CHECK(cudaFree(zeta_dev));
   GPU_ERROR_CHECK(cudaFree(zetb_dev));
   GPU_ERROR_CHECK(cudaFree(rpgfa_dev));
   GPU_ERROR_CHECK(cudaFree(rpgfb_dev));
   GPU_ERROR_CHECK(cudaFree(rab_dev));
   GPU_ERROR_CHECK(cudaFree(rr_dev));
   GPU_ERROR_CHECK(cudaFree(sab_dev));
}
*/


/*
   overlap integral v1, unoptmized
   sab : overlap matrix element over contracted Gaussian functions
   la_set, lb_set : angular momenta
   npgf_a, npgf_b : number of primitive Gaussian functions in contracted sets
   zet_a(1:npgf_a), zet_b(1:npgf_b) : Gaussian exponents
   gcc_a(1:npgf_a, 1:ncoa), gcc_b(1:npgf_b, 1:ncob) : Gaussian contracted coefficients for each primitive function and each Cartesian component

   Unlike Fortran, arrays' indicies in C start from 0.
*/
void overlap_ab_cgf_gpu_legacy(
   double *sab, int la_set, int npgf_a, const double *zet_a, const double *gcc_a,
   int lb_set, int npgf_b, const double *zet_b, const double *gcc_b, double rab_x, double rab_y, double rab_z)
{
   unsigned int ncoa = get_nco(la_set);
   unsigned int ncob = get_nco(lb_set);
   double *sab_pgf_dev = NULL;
   double *sab_dev = NULL;
   double *zet_a_dev = NULL;
   double *zet_b_dev = NULL;
   double *gcc_a_dev = NULL;
   double *gcc_b_dev = NULL;

   GPU_ERROR_CHECK(cudaMalloc( (void**) &sab_pgf_dev, npgf_a*npgf_b*ncoa*ncob*sizeof(double) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &sab_dev, ncoa*ncob*sizeof(double) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &zet_a_dev, npgf_a*sizeof(double) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &zet_b_dev, npgf_b*sizeof(double) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &gcc_a_dev, ncoa*npgf_a*sizeof(double) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &gcc_b_dev, ncob*npgf_b*sizeof(double) ));
   
   GPU_ERROR_CHECK(cudaMemcpy( zet_a_dev, zet_a, npgf_a*sizeof(double), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemcpy( zet_b_dev, zet_b, npgf_b*sizeof(double), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemcpy( gcc_a_dev, gcc_a, ncoa*npgf_a*sizeof(double), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemcpy( gcc_b_dev, gcc_b, ncob*npgf_b*sizeof(double), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemset( sab_dev, 0, ncoa*ncob*sizeof(double)));

   dim3 npgf_ab(npgf_a, npgf_b);
//   printf("A %d %d %d \n", npgf_a*npgf_b, ncoa, ncob);
   overlap_ab_cgf_kernel<<<1, npgf_ab >>>(
         sab_dev, sab_pgf_dev, gcc_a_dev, gcc_b_dev, zet_a_dev, zet_b_dev,
         la_set, lb_set, ncoa, ncob, rab_x, rab_y, rab_z );
   GPU_ERROR_CHECK(cudaGetLastError() );
   GPU_ERROR_CHECK(cudaMemcpy( sab, sab_dev, ncoa*ncob*sizeof(double), cudaMemcpyDeviceToHost ));
   GPU_ERROR_CHECK(cudaFree(zet_a_dev));
   GPU_ERROR_CHECK(cudaFree(zet_b_dev));
   GPU_ERROR_CHECK(cudaFree(gcc_a_dev));
   GPU_ERROR_CHECK(cudaFree(gcc_b_dev));
   GPU_ERROR_CHECK(cudaFree(sab_pgf_dev));
   GPU_ERROR_CHECK(cudaFree(sab_dev));
}

void overlap_ab_cgf(double *sab, int la_set, int npgf_a, const double *zet_a, const double *gcc_a, int lb_set, int npgf_b, const double *zet_b,
                    const double *gcc_b, double rab_x, double rab_y, double rab_z)
{
   unsigned int ncoa = get_nco(la_set);
   unsigned int ncob = get_nco(lb_set);
   double *sab_pgf = NULL;

   sab_pgf = (double*) malloc(ncoa*ncob*sizeof(*sab_pgf));
   if (sab_pgf == NULL) return;

   memset(sab, 0, ncoa*ncob*sizeof(*sab));

   for ( int ipgf_b = 0; ipgf_b < npgf_b; ++ipgf_b) {
      for ( int ipgf_a = 0; ipgf_a < npgf_a; ++ipgf_a) {
          if (la_set == 0 && lb_set == 0) {
             overlap_primitive_ss(sab_pgf, zet_a[ipgf_a], zet_b[ipgf_b], rab_x, rab_y, rab_z);
          } else if (la_set == 0 && lb_set == 1) {
             overlap_primitive_sp(sab_pgf, zet_a[ipgf_a], zet_b[ipgf_b], rab_x, rab_y, rab_z);
          } else if (la_set == 1 && lb_set == 0) {
             overlap_primitive_ps(sab_pgf, zet_a[ipgf_a], zet_b[ipgf_b], rab_x, rab_y, rab_z);
          } else if (la_set == 1 && lb_set == 1) {
             overlap_primitive_pp(sab_pgf, zet_a[ipgf_a], zet_b[ipgf_b], rab_x, rab_y, rab_z);
          }

          for (unsigned int icob = 0; icob < ncob; ++icob) {
             for (unsigned int icoa = 0; icoa < ncoa; ++icoa) {
                sab[icob*ncoa+icoa] += sab_pgf[icob*ncoa+icoa] * gcc_a[icoa*npgf_a+ipgf_a] * gcc_b[icob*npgf_b+ipgf_b];
             }
          }
      }
   }

   free(sab_pgf);
}

#define PAL_SLOTS 3
#define BAS_SLOTS 8
#define BAS_OFFSET_L 1
#define BAS_OFFSET_NPGF 2
#define BAS_OFFSET_Z 5
#define BAS_OFFSET_C 6
#define BAS_OFFSET_R 7

//   call compute_s ( list_ijd, atm, bas, env, s_sparse )
//  dim3 max_npgf_ab(max_npgf_col, mx_npgf_row)
//__global__ void compute_s_gpu<<< n_pairs, max_npgf_ab>>> ( int* list_ijd_dev, int* bas_dev, double* env_dev, double* s_sparse_dev )

__global__ void compute_s_gpu_kernel ( int* list_ijd_dev, int* bas_dev, double* env_dev, double* s_sparse_dev )
{
   int ijd_idx = blockIdx.x * PAL_SLOTS;
   int i = ( list_ijd_dev[ ijd_idx + 0 ] - 1 ) * BAS_SLOTS;
   int j = ( list_ijd_dev[ ijd_idx + 1 ] - 1 ) * BAS_SLOTS;
   int s_offset = list_ijd_dev[ ijd_idx + 2 ] - 1 ; // might be pushed to after the if, but it is more elegant here
   int ipgf_a = threadIdx.x;
   int ipgf_b = threadIdx.y;
   int npgf_a = bas_dev[i+BAS_OFFSET_NPGF];
   int npgf_b = bas_dev[j+BAS_OFFSET_NPGF];
   // We size the block to accomodate the largest contraction
   // so smaller contractions only use a subset of the threads
   // worse case is a contraction with high angular moment and a single coefficient
   // in which case one thread is doing all L calculations
   if ( (ipgf_a<npgf_a) and(ipgf_b<npgf_b)) {
      int la = bas_dev[i+BAS_OFFSET_L];
      int lb = bas_dev[j+BAS_OFFSET_L];
      int ncoa = get_nco(la);
      int ncob = get_nco(lb);
      double zet_a = env_dev[ bas_dev[i+BAS_OFFSET_Z] + ipgf_a ];
      double zet_b = env_dev[ bas_dev[j+BAS_OFFSET_Z] + ipgf_b ];
      double* gcc_a = &env_dev[ bas_dev[i+BAS_OFFSET_C] ];
      double* gcc_b = &env_dev[ bas_dev[j+BAS_OFFSET_C] ];
      double ra_x = env_dev[ bas_dev[i+BAS_OFFSET_R] + 0 ];
      double ra_y = env_dev[ bas_dev[i+BAS_OFFSET_R] + 1 ];
      double ra_z = env_dev[ bas_dev[i+BAS_OFFSET_R] + 2 ];
      double rb_x = env_dev[ bas_dev[j+BAS_OFFSET_R] + 0 ];
      double rb_y = env_dev[ bas_dev[j+BAS_OFFSET_R] + 1 ];
      double rb_z = env_dev[ bas_dev[j+BAS_OFFSET_R] + 2 ];
      double rab_x = ra_x - rb_x;
      double rab_y = ra_y - rb_y;
      double rab_z = ra_z - rb_z;
//      double sab_pgf_spher[9]; // nsoa*nsob]; // if L = 6, this is ((2*6+1)**2 = 169 doubles per thread. Also, this needs to be constant ( at compile time ?)
      double sab_pgf[9]; // ncoa*ncob]; // if L = 6, this is ((6+1)*(6+2)/2)**2 = 784 doubles per thread. Not great. Also, this needs to be constant ( at compile time ?)
      double s[16]; // ncoseta*ncosetb]; // if L = 6, this is [sum from 1 to 6 of ((l+1)*(l+2)/2)]**2 = 7056 doubles per thread. Not great. At all. Also, this needs to be constant ( at compile time ?)
      double cSc_ab;
      sab_pgf[0] = 0.0 ;
      sab_pgf[1] = 0.0 ;
      sab_pgf[2] = 0.0 ;
      sab_pgf[3] = 0.0 ;
      sab_pgf[4] = 0.0 ;
      sab_pgf[5] = 0.0 ;
      sab_pgf[6] = 0.0 ;
      sab_pgf[7] = 0.0 ;
      sab_pgf[8] = 0.0 ;

      //
      // Compute the gaussian integrals and saves them in sab_pgf
      if (la == 0 && lb == 0) {
         overlap_primitive_ss(s, zet_a, zet_b, rab_x, rab_y, rab_z);
      } else if (la == 0 && lb == 1) {
         overlap_primitive_sp(s, zet_a, zet_b, rab_x, rab_y, rab_z);
      } else if (la == 1 && lb == 0) {
         overlap_primitive_ps(s, zet_a, zet_b, rab_x, rab_y, rab_z);
      } else if (la == 1 && lb == 1) {
         overlap_primitive_pp(s, zet_a, zet_b, rab_x, rab_y, rab_z);
      }


//      overlap( s, zet_a, zet_b, rab_x, rab_y, rab_z, la, lb );

      if (la == 0 && lb == 0) {
         sab_pgf[0] = s[0];
      } else if (la == 0 && lb == 1) {
         sab_pgf[0] = s[1];
         sab_pgf[1] = s[2];
         sab_pgf[2] = s[3];
      } else if (la == 1 && lb == 0) {
         sab_pgf[0] = s[1];
         sab_pgf[1] = s[2];
         sab_pgf[2] = s[3];
      } else if (la == 1 && lb == 1) {
         sab_pgf[0] = s[5];
         sab_pgf[1] = s[6];
         sab_pgf[2] = s[7];
         sab_pgf[3] = s[9];
         sab_pgf[4] = s[10];
         sab_pgf[5] = s[11];
         sab_pgf[6] = s[13];
         sab_pgf[7] = s[14];
         sab_pgf[8] = s[15];
      }
//      printf("BlockIdx %d ThreadIdx %d ThreadIdy %d s_offset %d la %d lb %d  %e %e %e %e %e %e %e %e %e \n ", 
//              blockIdx.x, threadIdx.x, threadIdx.y, s_offset,   la,   lb,
//              sab_pgf[0],sab_pgf[1],sab_pgf[2],sab_pgf[3],sab_pgf[4],sab_pgf[5],sab_pgf[6],sab_pgf[7],sab_pgf[8]);

      // Contract the gaussian integrals to the different products between basis set functions
      for (unsigned int icob = 0; icob < ncob; ++icob) {
         for (unsigned int icoa = 0; icoa < ncoa; ++icoa) {
            cSc_ab = sab_pgf[icob*ncoa+icoa] *  gcc_a[icoa*npgf_a+ipgf_a] * gcc_b[icob*npgf_b+ipgf_b];
            // Thanks to s_offset, writes to sab_dev from different blocks will never overlap
            atomicAdd_block(&s_sparse_dev[s_offset + icob*ncoa+icoa ], cSc_ab);
         }
      }
   }
}

void compute_s_gpu ( int* list_ijd, int* bas, double* env, double* s_sparse,
                     int n_pairs,   int nbas, int env_size, int s_sparse_size,
                     int max_npgf_col, int max_npgf_row )
{
   int* list_ijd_dev = NULL;
   int* bas_dev = NULL;
   double* env_dev = NULL;
   double* s_sparse_dev = NULL;
   dim3 max_npgf_ab(max_npgf_col, max_npgf_row);

   // copy list of pairs and enviroment to gpu
   GPU_ERROR_CHECK(cudaMalloc( (void**) &list_ijd_dev, n_pairs * PAL_SLOTS * sizeof(int) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &bas_dev, nbas * BAS_SLOTS * sizeof(int) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &env_dev, env_size * sizeof(double) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &s_sparse_dev, s_sparse_size * sizeof(double) ));

   GPU_ERROR_CHECK(cudaMemcpy( list_ijd_dev, list_ijd, n_pairs * PAL_SLOTS * sizeof(int), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemcpy( bas_dev, bas, nbas * BAS_SLOTS * sizeof(int), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemcpy( env_dev, env, env_size * sizeof(double), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemset( s_sparse_dev, 0.0, s_sparse_size*sizeof(double)));
   // work
   compute_s_gpu_kernel<<< n_pairs, max_npgf_ab>>> ( list_ijd_dev, bas_dev, env_dev, s_sparse_dev );
   GPU_ERROR_CHECK(cudaGetLastError() );
   // copy back to ram and free memory
   GPU_ERROR_CHECK(cudaMemcpy( s_sparse, s_sparse_dev, s_sparse_size * sizeof(double), cudaMemcpyDeviceToHost ));
   GPU_ERROR_CHECK(cudaFree(list_ijd_dev));
   GPU_ERROR_CHECK(cudaFree(bas_dev));
   GPU_ERROR_CHECK(cudaFree(env_dev));
   GPU_ERROR_CHECK(cudaFree(s_sparse_dev));
}


void norm_cgf_gto(int l_set, int npgf, const double *zet, const double *gcc, double *gcc_total)
{
   double *sab = NULL;
   double zero = (double)0.0;
   double norm;
   unsigned int nco = get_nco(l_set);
   // sab(1:nco, 1:nco, 1:npgf, 1:npgf)
   sab = (double*) malloc(nco*nco*npgf*npgf*sizeof(*sab));
   if (sab == NULL) return;

   for (int ipgf = 0; ipgf < npgf; ++ipgf) {
      for (int jpgf = 0; jpgf < npgf; ++jpgf) {
          if (l_set == 0) {
             // sab(:, :, jpgf, ipgf)
             overlap_primitive_ss(sab+(ipgf*npgf+jpgf)*nco*nco, zet[jpgf], zet[ipgf], zero, zero, zero);
          } else if (l_set == 1) {
             overlap_primitive_pp(sab+(ipgf*npgf+jpgf)*nco*nco, zet[jpgf], zet[ipgf], zero, zero, zero);
          }
      }
   }

   for (unsigned int ico = 0; ico < nco; ++ico) {
      for (int ipgf = 0; ipgf < npgf; ++ipgf) {
         // sab(ico, ico, ipgf, ipgf)
         gcc_total[ico*npgf+ipgf] = gcc[ipgf] / sqrt(sab[((ipgf*npgf+ipgf)*nco+ico)*nco+ico]);
      }

      norm = (double)0.0;
      for ( int ipgf = 0; ipgf < npgf; ++ipgf) {
         for ( int jpgf = 0; jpgf < npgf; ++jpgf) {
             // sab(ico, ico, jpgf, ipgf)
             norm += sab[((ipgf*npgf+jpgf)*nco+ico)*nco+ico] * gcc_total[ico*npgf+jpgf] * gcc_total[ico*npgf+ipgf];
         }
      }

      norm = (double)1.0 / sqrt(norm);

      for ( int ipgf = 0; ipgf < npgf; ++ipgf) {
         gcc_total[ico*npgf+ipgf] = gcc_total[ico*npgf+ipgf] * norm;
      }
   }

   free(sab);
}


} // end of extern C
