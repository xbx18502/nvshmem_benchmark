#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L node=2
#PJM -L elapse=00:05:00
#PJM -j
#PJM -S


module purge
module load nvidia/24.11 nvhpcx/24.11-cuda12

export NVCC_GENCODE="arch=compute_90,code=sm_90"

export NVSHMEM_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/12.6/nvshmem"

export HYDRA_HOME="$NVSHMEM_HOME"

export CUDA_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6"

export MPI_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/12.6/hpcx/hpcx-2.20/ompi"
# NCCL settings
export NCCL_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/nccl"

export OMPI_MCA_plm_rsh_agent="/usr/bin/pjrsh"

export NVSHMEM_BOOTSTRAP=MPI

# -x NVSHMEMTEST_USE_MPI_LAUNCHER=1 \
task_mpi=" \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 2 --map-by ppr:1:node \
--bind-to numa   \
../bin/threadGroup.out"

profileWithNsys=" \
nsys profile --mpi-impl=openmpi -t cuda,nvtx -o mpi_init_put_bw_${PJM_JOBID}_${PJM_JOBID}.qdrep \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 8 --map-by ppr:4:node \
../bin/alltoall.out"

echo "command: ${task_mpi}"
for i in {1..1}
do
    echo "iteration: ${i}"
    eval ${task_mpi}
done