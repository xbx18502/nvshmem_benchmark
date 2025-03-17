#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L node=2
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

# export LD_LIBRARY_PATH="$NCCL_HOME/lib:$LD_LIBRARY_PATH"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$MPI_HOME/lib:$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH"

compile_dynamic=" \
nvcc -rdc=true -ccbin g++ -gencode=$NVCC_GENCODE -I \
$NVSHMEM_HOME/include Attribute-Based-Initialization-Example.cu -o \
Attribute-Based-Initialization-Example.out -L $NVSHMEM_HOME/lib -lnvshmem_host -lnvshmem_device "

compile_static=" \
nvcc -rdc=true -ccbin g++ -gencode=$NVCC_GENCODE -I \
$NVSHMEM_HOME/include:$MPI_HOME/include shmem_put_bw.cu -o \
shmem_put_bw.out -L $NVSHMEM_HOME/lib:$MPI_HOME/lib \
-lmpi -lnvshmem -lnvidia-ml -lcuda -lcudart "

compile2="nvcc -rdc=true -ccbin g++ -gencode=$NVCC_GENCODE \
-I$NVSHMEM_HOME/include -I$MPI_HOME/include \
Attribute-Based-Initialization-Example.cu \
-o Attribute-Based-Initialization-Example.out \
-L$NVSHMEM_HOME/lib -L$MPI_HOME/lib \
-lnvshmem -lmpi -lnvidia-ml -lcuda -lcudart"

export OMPI_MCA_plm_rsh_agent="/usr/bin/pjrsh"

task_nvshmrun=" \
$HYDRA_HOME/bin/nvshmrun -n 4 -ppn 4 \
-hostfile ${PJM_O_NODEINF} \
./Attribute-Based-Initialization-Example.out "

task_mpi=" \
mpirun -v --display-allocation --display-map -hostfile ${PJM_O_NODEINF} \
-np 8 --map-by ppr:4:node \
-x NVSHMEMTEST_USE_MPI_LAUNCHER=1 \
./Attribute-Based-Initialization-Example.out "

echo ${compile_static}
exec ${compile_static}
# echo ${task_nvshmrun}

# eval ${task_nvshmrun}

# echo ${task_mpi}

# eval ${task_mpi}
