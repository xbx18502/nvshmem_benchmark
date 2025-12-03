cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER "/home/app/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6/bin/nvcc")
set(CMAKE_CUDA_COMPILER_FORCED TRUE)
project(threadGroup CUDA CXX)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_COMPILER_FORCED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(NVCC_GENCODE "arch=compute_70,code=sm_70")
set(NVSHMEM_HOME "/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/12.6/nvshmem")
set(HYDRA_HOME "$NVSHMEM_HOME")
set(CUDA_HOME "/home/app/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6")
set(MPI_HOME "/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/12.6/hpcx/hpcx-2.20/ompi")
set(export NCCL_HOME "/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/nccl")


# Set paths
set(NVSHMEM_HOME "/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/12.6/nvshmem")
set(CUDA_HOME "/home/app/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6")
set(MPI_HOME "/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/12.6/hpcx/hpcx-2.20/ompi")
set(NCCL_HOME "/home/app/nvhpc/24.11/Linux_x86_64/24.11/comm_libs/nccl")
set(CUDAToolkit_ROOT "/home/app/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6")

# Add executable
add_executable(threadGroup 
    threadGroup.cu
)

install(TARGETS threadGroup
        RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)

# Enable device code linking
set_target_properties(threadGroup PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    OUTPUT_NAME "threadGroup.out"
)


# Set include directories
target_include_directories(threadGroup PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${NVSHMEM_HOME}/include
    ${MPI_HOME}/include
)

# Set link directories
target_link_directories(threadGroup PRIVATE
    ${NVSHMEM_HOME}/lib
    ${MPI_HOME}/lib
)

# Link libraries
target_link_libraries(threadGroup PRIVATE
    mpi
    nvshmem
    nvidia-ml
    cuda
    cudart
)