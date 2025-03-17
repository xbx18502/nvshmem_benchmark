#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:03:00
#PJM -j
#PJM -S


# module purge
# module load nvidia/24.11 nvhpcx/24.11-cuda12

export NVCC_GENCODE="arch=compute_90,code=sm_90"

export NVSHMEM_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/12.6/nvshmem"

export HYDRA_HOME="$NVSHMEM_HOME"

export CUDA_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6"

export MPI_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/12.6/openmpi4/openmpi-4.1.5"
# NCCL settings
export NCCL_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/nccl"

# export OMPI_MCA_plm_rsh_agent="/usr/bin/pjrsh"

export NVSHMEM_BOOTSTRAP=MPI

export LD_LIBRARY_PATH="$NCCL_HOME/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$MPI_HOME/lib:$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH"

# -x NVSHMEMTEST_USE_MPI_LAUNCHER=1 \
task_mpi=" \
$MPI_HOME/bin/mpirun -v --display-allocation --display-map \
-x NVSHMEMTEST_USE_MPI_LAUNCHER=1 \
-hostfile ${PJM_O_NODEINF} \
-np 4 --map-by ppr:4:node \
./fcollect.out "

for i in {1..1}
do
    echo "iteration: ${i}"
    eval ${task_mpi}
    echo " "
done