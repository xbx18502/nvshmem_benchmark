#!/bin/bash
export NVSHMEM_HOME="/home/app/nvhpc/23.9/Linux_x86_64/23.9/comm_libs/12.2/nvshmem"
export MPI_HOME='/home/app/gcc/8/ompi-cuda/4.1.6-12.2.2'
export NVCC_GENCODE="arch=compute_90,code=sm_90"

compile_static=" \
nvcc -rdc=true -ccbin g++ -gencode=$NVCC_GENCODE \
-I$NVSHMEM_HOME/include \
-I$MPI_HOME/include \
-I/home/pj24001791/ku40000464/osu-Micro-Benchmarks/osu-micro-benchmarks-7.5-1/c/util \
osu_nvshmem_put_bw.cu -o osu_nvshmem_put_bw.out \
-L$NVSHMEM_HOME/lib \
-L$MPI_HOME/lib \
-lmpi -lnvshmem -lnvidia-ml -lcuda -lcudart"


a="nvcc -o osu_nvshmem_put_bw osu_nvshmem_put_bw.cu -I$NVSHMEM_HOME/include -L<path_to_nvshmem>/lib -lnvshmem -L<path_to_mpi>/lib -lmpi -lcuda -lcudart"
exec ${compile_static}