#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:02:00
#PJM -j
#PJM -S


module purge
module load nvidia/23.9 nvhpcx/23.9-cuda12

export NVCC_GENCODE="arch=compute_90,code=sm_90"

export NVSHMEM_HOME="/home/app/nvhpc/23.9/Linux_x86_64/23.9/comm_libs/12.2/nvshmem"

export HYDRA_HOME="$NVSHMEM_HOME"

export CUDA_HOME="/home/app/nvhpc/23.9/Linux_x86_64/23.9/cuda/12.2"

export MPI_HOME="/home/app/nvhpc/23.9/Linux_x86_64/23.9/comm_libs/12.2/hpcx/hpcx-2.16/ompi"
# NCCL settings
export NCCL_HOME="/home/app/nvhpc/23.9/Linux_x86_64/23.9/comm_libs/nccl"

export OMPI_MCA_plm_rsh_agent="/usr/bin/pjrsh"


task_mpi=" \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 2 --map-by ppr:2:node \
-x NVSHMEMTEST_USE_MPI_LAUNCHER=1 \
./bw_test.out "

exec ${task_mpi}