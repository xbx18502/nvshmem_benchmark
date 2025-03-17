#include <cstddef>
#include <stdio.h>
#include "mpi.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include <iostream>
#include <assert.h>
#define MAX_MSG_SIZE (32 * 1024 * 1024)

#define MAX_ITERS 200
#define MAX_SKIP 20
#define BLOCKS 4
#define THREADS_PER_BLOCK 1024

#define CUDA_CHECK(stmt)                                  \
do {                                                      \
    cudaError_t result = (stmt);                          \
    if (cudaSuccess != result) {                          \
        fprintf(stderr, "[%s:%d] CUDA failed with %s \n", \
         __FILE__, __LINE__, cudaGetErrorString(result)); \
        exit(-1);                                         \
    }                                                     \
} while (0)

const int message_size = 1<<5;

int skip = 1000;
int loop = 10000;
int skip_large = 10;
int loop_large = 100;
int large_message_size = 8192;

__global__ void simple_shift(double *destination, int size) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;

    // nvshmem_int_p(destination, mype, peer);
    nvshmemx_double_put_nbi_block(destination,destination, size, peer);
}

__global__ void bw(double* dest, int size){
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);
    int bid = blockIdx.x;
    int nblocks = gridDim.x;
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;
    nvshmemx_double_put_nbi_block(dest + (bid * (size / nblocks)),
                                  dest + (bid * (size / nblocks)), size / nblocks, peer);
    nvshmem_quiet();

}

__global__ void bw_put(double* dest, int size){
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);
    int bid = blockIdx.x;
    int nblocks = gridDim.x;
    int nthreads = blockDim.x * blockDim.y * blockDim.z;
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;
    nvshmem_double_put_nbi(dest + (tid*(size/nthreads)), dest + (tid*(size/nthreads)), size/nthreads, peer);
    nvshmem_quiet();

}
int main (int argc, char *argv[]) {
    int mype_node;
    double* msg = (double*)malloc(sizeof(double)*message_size);
    double* msg_main = (double*)malloc(sizeof(double)*message_size);
    cudaStream_t stream;
    int rank, nranks;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr;
    std::cout<<"complete nvshmemx_init_attr_t attr"<<std::endl;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    std::cout<<"complete MPI_Init"<<std::endl;
    attr.mpi_comm = &mpi_comm;
    std::cout<<"complete attr.mpi_comm = &mpi_comm"<<std::endl;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    std::cout<<"complete nvshmemx_init_attr"<<std::endl;
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    std::cout<<"complete nvshmemx_init_attr"<<std::endl;
    CUDA_CHECK(cudaSetDevice(mype_node));
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::cout<<"complete cudaSetDevice"<<std::endl;
    double *destination = (double *) nvshmem_malloc (sizeof(double)*message_size);
    std::cout<<"complete nvshmem_malloc"<<std::endl;
    // CUDA_CHECK(cudaMemset(destination, 0, sizeof(int)*message_size));
    if(0==mype_node) {
        for(int i=0; i<message_size; i++) {
            msg_main[i] = mype_node + i;
        }
    }
    else{
        for(int i=0; i<message_size; i++) {
            msg_main[i] = mype_node + i;
        }
    }
    CUDA_CHECK(cudaMemcpy(destination, msg_main, sizeof(double)*message_size,
                cudaMemcpyHostToDevice));
    std::cout<<"complete *destination init"<<std::endl;
    #define MAX_MSG_SIZE_PT2PT (1 << 20)
    #define FLOAT_PRECISION 2
    int size;
        #define HEADER "# " "OSU OpenSHMEM Put Bandwidth Test" " v" "7.5" "\n"
    #define FIELD_WIDTH 18
    if (0 == mype_node) {
        fprintf(stdout, HEADER);
        fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH,
                "Bandwidth (MB/s)");
        fflush(stdout);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::cout<<"complete cudaEventCreate"<<std::endl;
    
    // nvshmemx_barrier_all_on_stream(stream);
    // std::cout<<"start = "<<start<<std::endl;
    // std::cout<<"stop = "<<stop<<std::endl;
    nvshmemx_barrier_all_on_stream(stream);
    
    bw_put<<<4,8, 0, stream>>>(destination, message_size);
    //bw<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(destination, message_size);
    // simple_shift<<<1, 1, 0, stream>>>(destination);
    // std::cout<<"complete simple_shift"<<std::endl;
    nvshmemx_barrier_all_on_stream(stream);
    std::cout<<"complete nvshmemx_barrier_all_on_stream"<<std::endl;
    CUDA_CHECK(cudaMemcpyAsync(msg, destination, sizeof(double)*message_size,
                 cudaMemcpyDeviceToHost, stream));
    std::cout<<"complete cudaMemcpyAsync"<<std::endl;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout<<"complete cudaStreamSynchronize"<<std::endl;
    
    printf("%d: initiliazed message ", nvshmem_my_pe());
    for(int i=0; i<message_size; i++) {
        printf("%f ", msg_main[i]);
    }
    printf("\n");

    printf("%d: received message ", nvshmem_my_pe());
    for(int i=0; i<message_size; i++) {
        printf("%f ", msg[i]);
    }
    printf("\n");

    nvshmem_free(destination);
    nvshmem_finalize();
    MPI_Finalize();
    return 0;
}
