module use /work4/scd/scarf562/eb-common/modules/all
module load amd-modules
module load OpenMPI/4.1.4-GCC-11.3.0
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.1
module load UCX-CUDA/1.12.1-GCCcore-11.3.0-CUDA-11.7.1
module load ScaLAPACK/2.2.0-gompi-2022a-fb

cd tools/toolchain/
./install_cp2k_toolchain.sh
wait
cd ../../

