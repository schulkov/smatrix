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

__host__ __device__ inline unsigned int get_nco(int l)
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
void overlap_ab_cgf_gpu_legacy(
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
//   printf("A %d %d %d \n", npgf_a*npgf_b, ncoa, ncob);
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

void overlap_ab_cgf(REAL_T *sab, int la_set, int npgf_a, const REAL_T *zet_a, const REAL_T *gcc_a, int lb_set, int npgf_b, const REAL_T *zet_b,
                    const REAL_T *gcc_b, REAL_T rab_x, REAL_T rab_y, REAL_T rab_z)
{
   unsigned int ncoa = get_nco(la_set);
   unsigned int ncob = get_nco(lb_set);
   REAL_T *sab_pgf = NULL;

   sab_pgf = (REAL_T*) malloc(ncoa*ncob*sizeof(*sab_pgf));
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
   // We size the block to accomodate the largest contractionso smaller contractions only use a subset of the threads
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
      double sab_pgf[9]; // ncoa*ncob]; // if L = 6, this is ((6+1)*(6+2)/2)**2 = 784 doubles per thread. Not great. Also, this needs to be constant ( at compile time ?)
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
         overlap_primitive_ss(&sab_pgf[0], zet_a, zet_b, rab_x, rab_y, rab_z);
      } else if (la == 0 && lb == 1) {
         overlap_primitive_sp(sab_pgf, zet_a, zet_b, rab_x, rab_y, rab_z);
      } else if (la == 1 && lb == 0) {
         overlap_primitive_ps(sab_pgf, zet_a, zet_b, rab_x, rab_y, rab_z);
      } else if (la == 1 && lb == 1) {
         overlap_primitive_pp(sab_pgf, zet_a, zet_b, rab_x, rab_y, rab_z);
      }

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

void compute_s_gpu ( int* list_ijd, int* bas, REAL_T* env, REAL_T* s_sparse,
                     int n_pairs,   int nbas, int env_size, int s_sparse_size,
                     int max_npgf_col, int max_npgf_row )
{
   int* list_ijd_dev = NULL;
   int* bas_dev = NULL;
   REAL_T* env_dev = NULL;
   REAL_T* s_sparse_dev = NULL;
   dim3 max_npgf_ab(max_npgf_col, max_npgf_row);

   // copy list of pairs and enviroment to gpu
   GPU_ERROR_CHECK(cudaMalloc( (void**) &list_ijd_dev, n_pairs * PAL_SLOTS * sizeof(int) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &bas_dev, nbas * BAS_SLOTS * sizeof(int) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &env_dev, env_size * sizeof(REAL_T) ));
   GPU_ERROR_CHECK(cudaMalloc( (void**) &s_sparse_dev, s_sparse_size * sizeof(REAL_T) ));

   GPU_ERROR_CHECK(cudaMemcpy( list_ijd_dev, list_ijd, n_pairs * PAL_SLOTS * sizeof(int), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemcpy( bas_dev, bas, nbas * BAS_SLOTS * sizeof(int), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemcpy( env_dev, env, env_size * sizeof(REAL_T), cudaMemcpyHostToDevice ));
   GPU_ERROR_CHECK(cudaMemset( s_sparse_dev, 0.0, s_sparse_size*sizeof(REAL_T)));
   // work
   compute_s_gpu_kernel<<< n_pairs, max_npgf_ab>>> ( list_ijd_dev, bas_dev, env_dev, s_sparse_dev );
   GPU_ERROR_CHECK(cudaGetLastError() );
   // copy back to ram and free memory
   GPU_ERROR_CHECK(cudaMemcpy( s_sparse, s_sparse_dev, s_sparse_size * sizeof(REAL_T), cudaMemcpyDeviceToHost ));
   GPU_ERROR_CHECK(cudaFree(list_ijd_dev));
   GPU_ERROR_CHECK(cudaFree(bas_dev));
   GPU_ERROR_CHECK(cudaFree(env_dev));
   GPU_ERROR_CHECK(cudaFree(s_sparse_dev));
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

      norm = (REAL_T)0.0;
      for ( int ipgf = 0; ipgf < npgf; ++ipgf) {
         for ( int jpgf = 0; jpgf < npgf; ++jpgf) {
             // sab(ico, ico, jpgf, ipgf)
             norm += sab[((ipgf*npgf+jpgf)*nco+ico)*nco+ico] * gcc_total[ico*npgf+jpgf] * gcc_total[ico*npgf+ipgf];
         }
      }

      norm = (REAL_T)1.0 / sqrt(norm);

      for ( int ipgf = 0; ipgf < npgf; ++ipgf) {
         gcc_total[ico*npgf+ipgf] = gcc_total[ico*npgf+ipgf] * norm;
      }
   }

   free(sab);
}


} // end of extern C
