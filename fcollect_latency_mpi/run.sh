#!/bin/bash
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:03:00
#PJM -j
#PJM -S



module load cuda/12.2.2 nccl/2.22.3 gcc/8 ompi-cuda/4.1.6-12.2.2

# /home/app/nvhpc/24.11/Linux_x86_64/24.11/profilers/Nsight_Systems/bin/nsys profile --mpi-impl=openmpi -t cuda,nvtx -o mpi_init_put_bw_${PJM_JOBID}_${PJM_JOBID}.qdrep \

profile_task=" \
/home/app/nvhpc/24.11/Linux_x86_64/24.11/profilers/Nsight_Systems/bin/nsys profile --mpi-impl=openmpi -t cuda,nvtx -o mpi_init_put_bw_${PJM_JOBID}_${PJM_JOBID}.qdrep \
mpirun -np 4 \
--display-map --display-allocation -v \
--map-by socket --bind-to socket   \
/home/pj24001791/ku40000464/osu-Micro-Benchmarks/osu-nccl-install-path/libexec/osu-micro-benchmarks/mpi/collective/osu_gather \
"

eval $profile_task


