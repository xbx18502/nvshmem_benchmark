#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L node=1
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
export NVSHMEMTEST_MPI_SUPPORT=1

# -x NVSHMEMTEST_USE_MPI_LAUNCHER=1 
# --map-by socket --bind-to socket
task_mpi=" \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 4 --map-by ppr:4:node \
--bind-to numa \
-x NVSHMEMTEST_USE_MPI_LAUNCHER=1 \
../bin/broadcast_simple.out"

profileWithNsys=" \
nsys profile --mpi-impl=openmpi -t cuda,nvtx -o mpi_init_put_bw_${PJM_JOBID}_${PJM_JOBID}.qdrep \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 8 --map-by ppr:4:node \
../broadcast/broadcast.out "

echo "command: ${task_mpi}"
echo "node = ${PJM_O_NODEINF}"
echo "root = 0"
for i in {1..1}
do
    echo "iteration: ${i}"
    eval ${task_mpi}
done