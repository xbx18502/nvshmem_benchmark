module purge
module load nvidia/24.11 nvhpcx/24.11-cuda12

export NVCC_GENCODE="arch=compute_90,code=sm_90"

export NVSHMEM_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/12.6/nvshmem"

export HYDRA_HOME="$NVSHMEM_HOME"

export CUDA_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6"

export MPI_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/12.6/hpcx/hpcx-2.20/ompi"
# NCCL settings
export NCCL_HOME="/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/nccl"

compile_static=" \
nvcc -rdc=true -ccbin g++ -gencode=$NVCC_GENCODE -I \
$NVSHMEM_HOME/include:$MPI_HOME/include iterate_inside_kernel.cu -o \
iterate_inside_kernel.out -L $NVSHMEM_HOME/lib:$MPI_HOME/lib \
-lmpi -lnvshmem -lnvidia-ml -lcuda -lcudart "

echo ${compile_static}
eval ${compile_static}