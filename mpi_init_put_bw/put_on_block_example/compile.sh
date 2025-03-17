module purge
module load nvidia/23.9 nvhpcx/23.9-cuda12

export NVCC_GENCODE="arch=compute_90,code=sm_90"

export NVSHMEM_HOME="/home/app/nvhpc/23.9/Linux_x86_64/23.9/comm_libs/12.2/nvshmem"

export HYDRA_HOME="$NVSHMEM_HOME"

export CUDA_HOME="/home/app/nvhpc/23.9/Linux_x86_64/23.9/cuda/12.2"

export MPI_HOME="/home/app/nvhpc/23.9/Linux_x86_64/23.9/comm_libs/12.2/hpcx/hpcx-2.16/ompi"
# NCCL settings
export NCCL_HOME="/home/app/nvhpc/23.9/Linux_x86_64/23.9/comm_libs/nccl"

compile_static=" \
nvcc -rdc=true -ccbin g++ -gencode=$NVCC_GENCODE -I \
$NVSHMEM_HOME/include:$MPI_HOME/include o1.cu -o \
o1.out -L $NVSHMEM_HOME/lib:$MPI_HOME/lib \
-lmpi -lnvshmem -lnvidia-ml -lcuda -lcudart "

echo ${compile_static}
eval ${compile_static}