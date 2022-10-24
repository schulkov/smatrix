/* exp(), sqrt() */
#include <math.h>
/* malloc(), free() */
#include <stdlib.h>
/* memset() */
#include <string.h>
#include <stdio.h>
/* a way to switch precision : single <-> double */
typedef double REAL_T;


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

__host__ __device__ void overlap_primitive_ss(REAL_T *sab, REAL_T zeta, REAL_T zetb, REAL_T rab_x, REAL_T rab_y, REAL_T rab_z)
{
   REAL_T dab = sqrt(rab_x*rab_x + rab_y*rab_y + rab_z*rab_z);

   //  *** Prefactors ***
   REAL_T zetp = ((REAL_T)1.0)/(zeta+zetb);
   REAL_T pi_zetp = M_PI * zetp;
   REAL_T f0 = pi_zetp*sqrt(pi_zetp);
   REAL_T f1 = zetb*zetp;

   *sab = f0*exp(-zeta*f1*dab*dab);
}

void overlap_primitive_ss_cpu(REAL_T *sab, REAL_T zeta, REAL_T zetb, REAL_T rab_x, REAL_T rab_y, REAL_T rab_z)
{
   REAL_T dab = sqrt(rab_x*rab_x + rab_y*rab_y + rab_z*rab_z);

   //  *** Prefactors ***
   REAL_T zetp = ((REAL_T)1.0)/(zeta+zetb);
   REAL_T pi_zetp = M_PI * zetp;
   REAL_T f0 = pi_zetp*sqrt(pi_zetp);
   REAL_T f1 = zetb*zetp;

   *sab = f0*exp(-zeta*f1*dab*dab);
}


__host__ __device__ void overlap_primitive_sp(REAL_T *sab, REAL_T zeta, REAL_T zetb, REAL_T rab_x, REAL_T rab_y, REAL_T rab_z)
{
   REAL_T dab = sqrt(rab_x*rab_x + rab_y*rab_y + rab_z*rab_z);

   //  *** Prefactors ***
   REAL_T zetp = ((REAL_T)1.0)/(zeta+zetb);
   REAL_T pi_zetp = M_PI * zetp;
   REAL_T f0 = sqrt(pi_zetp*pi_zetp*pi_zetp);
   REAL_T f1 = zetb*zetp;
   REAL_T f1_m1 = f1 - (REAL_T)1.0;
   REAL_T rbp_x = f1_m1*rab_x;
   REAL_T rbp_y = f1_m1*rab_y;
   REAL_T rbp_z = f1_m1*rab_z;

   REAL_T s0 = f0*exp(-zeta*f1*dab*dab); // [s|s]
   sab[0] = rbp_x*s0; // [s|px]
   sab[1] = rbp_y*s0; // [s|py]
   sab[2] = rbp_z*s0; // [s|pz]
}

/* the same as overlap_primitive_sp(sab, zetb, zeta, rab_x, rab_y, rab_z) due to symmetry */
__host__ __device__ void overlap_primitive_ps(REAL_T *sab, REAL_T zeta, REAL_T zetb, REAL_T rab_x, REAL_T rab_y, REAL_T rab_z)
{
   REAL_T dab = sqrt(rab_x*rab_x + rab_y*rab_y + rab_z*rab_z);

   //  *** Prefactors ***
   REAL_T zetp = ((REAL_T)1.0)/(zeta+zetb);
   REAL_T pi_zetp = M_PI * zetp;
   REAL_T f0 = sqrt(pi_zetp*pi_zetp*pi_zetp);
   REAL_T f1 = zetb*zetp;
   REAL_T rap_x = f1*rab_x;
   REAL_T rap_y = f1*rab_y;
   REAL_T rap_z = f1*rab_z;

   REAL_T s0 = f0*exp(-zeta*f1*dab*dab); // [s|s]
   sab[0] = rap_x*s0; // [px|s]
   sab[1] = rap_y*s0; // [py|s]
   sab[2] = rap_z*s0; // [pz|s]
}

void overlap_primitive_pp_cpu(REAL_T *sab, REAL_T zeta, REAL_T zetb, REAL_T rab_x, REAL_T rab_y, REAL_T rab_z)
{
   REAL_T dab = sqrt(rab_x*rab_x + rab_y*rab_y + rab_z*rab_z);

   //  *** Prefactors ***
   REAL_T zetp = ((REAL_T)1.0)/(zeta+zetb);
   REAL_T pi_zetp = M_PI * zetp;
   REAL_T f0 = sqrt(pi_zetp*pi_zetp*pi_zetp);
   REAL_T f1 = zetb*zetp;
   REAL_T f2 = ((REAL_T)0.5)*zetp;
   REAL_T rap_x = f1*rab_x;
   REAL_T rap_y = f1*rab_y;
   REAL_T rap_z = f1*rab_z;
   REAL_T rbp_x = rap_x-rab_x;
   REAL_T rbp_y = rap_y-rab_y;
   REAL_T rbp_z = rap_z-rab_z;
   REAL_T s0, s1;

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


__host__ __device__ void overlap_primitive_pp(REAL_T *sab, REAL_T zeta, REAL_T zetb, REAL_T rab_x, REAL_T rab_y, REAL_T rab_z)
{
   REAL_T dab = sqrt(rab_x*rab_x + rab_y*rab_y + rab_z*rab_z);

   //  *** Prefactors ***
   REAL_T zetp = ((REAL_T)1.0)/(zeta+zetb);
   REAL_T pi_zetp = M_PI * zetp;
   REAL_T f0 = sqrt(pi_zetp*pi_zetp*pi_zetp);
   REAL_T f1 = zetb*zetp;
   REAL_T f2 = ((REAL_T)0.5)*zetp;
   REAL_T rap_x = f1*rab_x;
   REAL_T rap_y = f1*rab_y;
   REAL_T rap_z = f1*rab_z;
   REAL_T rbp_x = rap_x-rab_x;
   REAL_T rbp_y = rap_y-rab_y;
   REAL_T rbp_z = rap_z-rab_z;
   REAL_T s0, s1;

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

inline unsigned int get_nco(int l)
{
   unsigned int nco = 0;

   if (l >= 0) nco = (l+1)*(l+2)/2;
   return nco;
}


__global__ void overlap_ab_cgf_kernel(
    REAL_T* sab_dev, REAL_T* sab_pgf_dev, REAL_T* gcc_a_dev, REAL_T* gcc_b_dev, REAL_T* zet_a_dev, REAL_T* zet_b_dev,
    int la_set, int lb_set, unsigned int ncoa, unsigned int ncob, REAL_T rab_x, REAL_T rab_y, REAL_T rab_z )
{
   unsigned int ipgf_a = threadIdx.x ;
   unsigned int ipgf_b = threadIdx.y ;
   int npgf_a = blockDim.x;
   int npgf_b = blockDim.y;
   REAL_T gccSgcc_ab;
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
//         sab_dev[icob*ncoa+icoa] += gccSgcc_ab;
         atomicAdd_block(&sab_dev[icob*ncoa+icoa], gccSgcc_ab);
      }
  }
}



extern "C" {
/*
   overlap integral v1, unoptmized
   sab : overlap matrix element over contracted Gaussian functions
   la_set, lb_set : angular momenta
   npgf_a, npgf_b : number of primitive Gaussian functions in contracted sets
   zet_a(1:npgf_a), zet_b(1:npgf_b) : Gaussian exponents
   gcc_a(1:npgf_a, 1:ncoa), gcc_b(1:npgf_b, 1:ncob) : Gaussian contracted coefficients for each primitive function and each Cartesian component

   Unlike Fortran, arrays' indicies in C start from 0.
*/
void overlap_ab_cgf(
   REAL_T *sab, int la_set, int npgf_a, const REAL_T *zet_a, const REAL_T *gcc_a,
   int lb_set, int npgf_b, const REAL_T *zet_b, const REAL_T *gcc_b, REAL_T rab_x, REAL_T rab_y, REAL_T rab_z)
{
   unsigned int ncoa = get_nco(la_set);
   unsigned int ncob = get_nco(lb_set);
   REAL_T *sab_pgf_dev = NULL;
   REAL_T *sab_dev = NULL;
   REAL_T *zet_a_dev = NULL;
   REAL_T *zet_b_dev = NULL;
   REAL_T *gcc_a_dev = NULL;
   REAL_T *gcc_b_dev = NULL;

   GPU_ERROR_CHECK(cudaMalloc( (void**) &sab_pgf_dev, npgf_a*npgf_b*ncoa*ncob*sizeof(REAL_T) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &sab_dev, ncoa*ncob*sizeof(REAL_T) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &zet_a_dev, npgf_a*sizeof(REAL_T) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &zet_b_dev, npgf_b*sizeof(REAL_T) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &gcc_a_dev, ncoa*npgf_a*sizeof(REAL_T) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &gcc_b_dev, ncob*npgf_b*sizeof(REAL_T) ));
   
   GPU_ERROR_CHECK(cudaMemcpy( zet_a_dev, zet_a, npgf_a*sizeof(REAL_T), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemcpy( zet_b_dev, zet_b, npgf_b*sizeof(REAL_T), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemcpy( gcc_a_dev, gcc_a, ncoa*npgf_a*sizeof(REAL_T), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemcpy( gcc_b_dev, gcc_b, ncob*npgf_b*sizeof(REAL_T), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemset( sab_dev, 0, ncoa*ncob*sizeof(REAL_T)));

   dim3 npgf_ab(npgf_a, npgf_b);
   overlap_ab_cgf_kernel<<<1, npgf_ab >>>(
         sab_dev, sab_pgf_dev, gcc_a_dev, gcc_b_dev, zet_a_dev, zet_b_dev,
         la_set, lb_set, ncoa, ncob, rab_x, rab_y, rab_z );
   GPU_ERROR_CHECK(cudaGetLastError() );
   GPU_ERROR_CHECK(cudaMemcpy( sab, sab_dev, ncoa*ncob*sizeof(REAL_T), cudaMemcpyDeviceToHost ));
   GPU_ERROR_CHECK(cudaFree(zet_a_dev));
   GPU_ERROR_CHECK(cudaFree(zet_b_dev));
   GPU_ERROR_CHECK(cudaFree(gcc_a_dev));
   GPU_ERROR_CHECK(cudaFree(gcc_b_dev));
   GPU_ERROR_CHECK(cudaFree(sab_pgf_dev));
   GPU_ERROR_CHECK(cudaFree(sab_dev));
}


void norm_cgf_gto(int l_set, int npgf, const REAL_T *zet, const REAL_T *gcc, REAL_T *gcc_total)
{

   unsigned int nco = get_nco(l_set);
   REAL_T *sab = NULL;
   REAL_T zero = (REAL_T)0.0;
   REAL_T norm;

   // sab(1:nco, 1:nco, 1:npgf, 1:npgf)
   sab = (REAL_T*) malloc(nco*nco*npgf*npgf*sizeof(*sab));
   if (sab == NULL) return;

   for (unsigned int ipgf = 0; ipgf < npgf; ++ipgf) {
      for (unsigned int jpgf = 0; jpgf < npgf; ++jpgf) {
          if (l_set == 0) {
             // sab(:, :, jpgf, ipgf)
             overlap_primitive_ss_cpu(sab+(ipgf*npgf+jpgf)*nco*nco, zet[jpgf], zet[ipgf], zero, zero, zero);
          } else if (l_set == 1) {
             overlap_primitive_pp_cpu(sab+(ipgf*npgf+jpgf)*nco*nco, zet[jpgf], zet[ipgf], zero, zero, zero);
          }
      }
   }

   for (unsigned int ico = 0; ico < nco; ++ico) {
      for (unsigned int ipgf = 0; ipgf < npgf; ++ipgf) {
         // sab(ico, ico, ipgf, ipgf)
         gcc_total[ico*npgf+ipgf] = gcc[ipgf] / sqrt(sab[((ipgf*npgf+ipgf)*nco+ico)*nco+ico]);
      }

      norm = (REAL_T)0.0;
      for (unsigned int ipgf = 0; ipgf < npgf; ++ipgf) {
         for (unsigned int jpgf = 0; jpgf < npgf; ++jpgf) {
             // sab(ico, ico, jpgf, ipgf)
             norm += sab[((ipgf*npgf+jpgf)*nco+ico)*nco+ico] * gcc_total[ico*npgf+jpgf] * gcc_total[ico*npgf+ipgf];
         }
      }

      norm = (REAL_T)1.0 / sqrt(norm);

      for (unsigned int ipgf = 0; ipgf < npgf; ++ipgf) {
         gcc_total[ico*npgf+ipgf] = gcc_total[ico*npgf+ipgf] * norm;
      }
   }

   free(sab);
}
}
