#include <cstddef>
#include <stdio.h>
#include "host/nvshmem_api.h"
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

const int message_size = 1<<22;

int skip = 20;
int loop = 200;
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

}

__global__ void latency_block_nbi(double *data_d, volatile unsigned int *counter_d, int len, int pe, int iter) {
    int i, peer;
    unsigned int counter;
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);
    int bid = blockIdx.x;
    int nblocks = gridDim.x;

    peer = !pe;
    for (i = 0; i < iter; i++) {
        nvshmemx_double_put_nbi_block(data_d + (bid * (len / nblocks)),
                                      data_d + (bid * (len / nblocks)), len / nblocks, peer);

        // synchronizing across blocks
        __syncthreads();
        if (!tid) {
            __threadfence();
            counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
            if (counter == (gridDim.x * (i + 1) - 1)) {
                *(counter_d + 1) += 1;
            }
            while (*(counter_d + 1) != i + 1)
                ;
        }
        __syncthreads();
    }

    // synchronize and call nvshme_quiet
    __syncthreads();
    if (!tid) {
        __threadfence();
        counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
        if (counter == (gridDim.x * (i + 1) - 1)) {
            nvshmem_quiet();
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != i + 1)
            ;
    }
    __syncthreads();
}

__global__ void latency_warp_nbi(double *data_d, volatile unsigned int *counter_d, int len, int pe, int iter) {
    int i, peer;
    unsigned int counter;
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);
    int bid = blockIdx.x;
    int wid = tid / 32;
    int nblocks = gridDim.x;
    int nwarps = blockDim.x / 32;
    int total_warps = nwarps * nblocks;

    peer = !pe;
    for (i = 0; i < iter; i++) {
        nvshmemx_double_put_nbi_warp(
            data_d + (bid * nwarps + wid) * (len / total_warps),
            data_d + (bid * nwarps + wid) * (len / total_warps), 
            len / total_warps, 
            peer);

        __syncwarp();
        if (tid % 32 == 0) {
            __threadfence();
            counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
            if (counter == (total_warps * (i + 1) - 1)) {
                *(counter_d + 1) += 1;
            }
            while (*(counter_d + 1) != i + 1)
                ;
        }
        __syncthreads();
    }

    __syncwarp();
    if (tid % 32 == 0) {
        __threadfence();
        counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
        if (counter == (total_warps * (i + 1) - 1)) {
            nvshmem_quiet();
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != i + 1)
            ;
    }
    __syncthreads();
}

__global__ void latency_block(double *data_d, int len, int pe, int iter) { 
    int i, tid, peer;                                                     
                                                                            
    peer = !pe;                                                           
    tid = threadIdx.x;                                                    
                                                                            
    for (i = 0; i < iter; i++) {                                          
        nvshmemx_double_put_block(data_d, data_d, len, peer);              
                                                                            
        __syncthreads();                                                  
        if (!tid) nvshmem_quiet();                                        
        __syncthreads();                                                  
    }                                                                     
}
__global__ void latency_warp(double *data_d, int len, int pe, int iter) { 
    int i, tid, peer;                                                     
                                                                            
    peer = !pe;                                                           
    tid = threadIdx.x;                                                    
                                                                            
    for (i = 0; i < iter; i++) {                                          
        nvshmemx_double_put_warp(data_d, data_d, len, peer);              
                                                                            
        __syncthreads();                                                  
        if (!tid) nvshmem_quiet();                                        
        __syncthreads();                                                  
    }                                                                     
}
__global__ void latency(double *data_d, int len, int pe, int iter) {
    int i, peer;

    peer = !pe;

    for (i = 0; i < iter; i++) {
        nvshmem_double_put_nbi(data_d, data_d, len, peer);
        nvshmem_quiet();
    }
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
    for(int i=0; i<message_size; i++) {
        msg_main[i] = 42;
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
    unsigned int *counter_d;
    CUDA_CHECK(cudaMalloc((void **)&counter_d, sizeof(unsigned int) * 2));
    std::cout<<"init counter_d"<<std::endl;

    fprintf(stdout, "simple put latency:\n\n");
    for (size = 1; size <= MAX_MSG_SIZE_PT2PT; size = (size ? size * 2 : 1)) {
        if (size > large_message_size) {
            loop = loop_large = 100;
            skip = skip_large = 0;
        }
        //nvshmemx_barrier_all_on_stream(stream);
        //nvshmem_barrier_all();
        if (0 == mype_node) {
            
            
            CUDA_CHECK(cudaDeviceSynchronize());
            //-------bw2-------
            // CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            // bw2<<<4,1024, 0 ,stream>>> (destination, counter_d, size, !mype_node,skip);
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            // CUDA_CHECK(cudaStreamSynchronize(stream));
            latency<<<1,1>>>(destination, size, mype_node, skip);
            cudaEventRecord(start);
            //CUDA_CHECK(cudaEventSynchronize(start));

            // bw2<<<4,1024>>> (destination, counter_d, size, mype_node,loop);
            latency<<<1,1>>>(destination, size, mype_node, loop);
            //---------------------
            cudaEventRecord(stop);
            CUDA_CHECK(cudaEventSynchronize(stop));
            //CUDA_CHECK(cudaDeviceSynchronize());
            nvshmem_barrier_all();
        }
        else{
            // nvshmem_barrier_all();
            nvshmem_barrier_all();
            
        }
        //nvshmem_barrier_all();
        // CUDA_CHECK(cudaStreamSynchronize(stream));
        // nvshmemx_barrier_all_on_stream(stream);
        double mb_total = 0.0;
        double t_total = 0.0;
        float milliseconds = 0.0;
        if (0 == mype_node) {
            mb_total = size * loop *8/ ( 1e6);
            cudaEventElapsedTime(&milliseconds, start, stop);
            t_total = milliseconds/1e3*1e6;
            double bw = mb_total / t_total;
            double latency = (1.0 * (t_total) / loop);
            fprintf(stdout, "%-*d%*.*f\n", 10, size*8, FIELD_WIDTH,
                    FLOAT_PRECISION, latency);
            fflush(stdout);
            //std::cout<<"PE0 finish print"<<std::endl;
        }
        else{
            std::cout<<"PE1 finish print"<<std::endl;
        }
    }
    fprintf(stdout, "warp:\n\n");
    for (size = 1; size <= MAX_MSG_SIZE_PT2PT; size = (size ? size * 2 : 1)) {
        if (size > large_message_size) {
            loop = loop_large = 100;
            skip = skip_large = 0;
        }
        //nvshmemx_barrier_all_on_stream(stream);
        //nvshmem_barrier_all();
        if (0 == mype_node) {
            
            
            CUDA_CHECK(cudaDeviceSynchronize());
            //-------bw2-------
            // CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            // bw2<<<4,1024, 0 ,stream>>> (destination, counter_d, size, !mype_node,skip);
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            // CUDA_CHECK(cudaStreamSynchronize(stream));
            latency_warp_nbi<<<1,32>>>(destination, counter_d,size, mype_node, skip);
            cudaEventRecord(start);
            //CUDA_CHECK(cudaEventSynchronize(start));
            // bw2<<<4,1024>>> (destination, counter_d, size, mype_node,loop);
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            latency_warp_nbi<<<1, 32>>>(destination, counter_d,size, mype_node, loop);
            //---------------------
            cudaEventRecord(stop);
            CUDA_CHECK(cudaEventSynchronize(stop));
            //CUDA_CHECK(cudaDeviceSynchronize());
            nvshmem_barrier_all();
        }
        else{
            // nvshmem_barrier_all();
            nvshmem_barrier_all();
            
        }
        //nvshmem_barrier_all();
        // CUDA_CHECK(cudaStreamSynchronize(stream));
        // nvshmemx_barrier_all_on_stream(stream);
        double mb_total = 0.0;
        double t_total = 0.0;
        float milliseconds = 0.0;
        if (0 == mype_node) {
            mb_total = size * loop *8/ ( 1e6);
            cudaEventElapsedTime(&milliseconds, start, stop);
            t_total = milliseconds/1e3*1e6;
            double bw = mb_total / t_total;
            double latency = (1.0 * (t_total) / loop);
            fprintf(stdout, "%-*d%*.*f\n", 10, size*8, FIELD_WIDTH,
                    FLOAT_PRECISION, latency);
            fflush(stdout);
            //std::cout<<"PE0 finish print"<<std::endl;
        }
        else{
            std::cout<<"PE1 finish print"<<std::endl;
        }
    }
    
    fprintf(stdout, "block:\n\n");
    for (size = 1; size <= MAX_MSG_SIZE_PT2PT; size = (size ? size * 2 : 1)) {
        if (size > large_message_size) {
            loop = loop_large = 100;
            skip = skip_large = 0;
        }
        //nvshmemx_barrier_all_on_stream(stream);
        //nvshmem_barrier_all();
        if (0 == mype_node) {
            
            
            CUDA_CHECK(cudaDeviceSynchronize());
            //-------bw2-------
            // CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            // bw2<<<4,1024, 0 ,stream>>> (destination, counter_d, size, !mype_node,skip);
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            // CUDA_CHECK(cudaStreamSynchronize(stream));
            latency_block_nbi<<<1, 1024>>>(destination, counter_d,size, mype_node, skip);
            cudaEventRecord(start);
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            //CUDA_CHECK(cudaEventSynchronize(start));
            // bw2<<<4,1024>>> (destination, counter_d, size, mype_node,loop);
            latency_block_nbi<<<1, 1024>>>(destination, counter_d,size, mype_node, loop);
            //---------------------
            cudaEventRecord(stop);
            CUDA_CHECK(cudaEventSynchronize(stop));
            //CUDA_CHECK(cudaDeviceSynchronize());
            nvshmem_barrier_all();
        }
        else{
            // nvshmem_barrier_all();
            nvshmem_barrier_all();
            
        }
        //nvshmem_barrier_all();
        // CUDA_CHECK(cudaStreamSynchronize(stream));
        // nvshmemx_barrier_all_on_stream(stream);
        double mb_total = 0.0;
        double t_total = 0.0;
        float milliseconds = 0.0;
        if (0 == mype_node) {
            mb_total = size * loop *8/ ( 1e6);
            cudaEventElapsedTime(&milliseconds, start, stop);
            t_total = milliseconds/1e3*1e6;
            double bw = mb_total / t_total;
            double latency = (1.0 * (t_total) / loop);
            fprintf(stdout, "%-*d%*.*f\n", 10, size*8, FIELD_WIDTH,
                    FLOAT_PRECISION, latency);
            fflush(stdout);
            //std::cout<<"PE0 finish print"<<std::endl;
        }
        else{
            std::cout<<"PE1 finish print"<<std::endl;
        }
    }
    std::cout<<"finish the loop"<<std::endl;
    cudaEventDestroy(start);
    std::cout<<"finish destroy the start"<<std::endl;
    cudaEventDestroy(stop);
    std::cout<<"finish destroy the stop"<<std::endl;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvshmem_free(destination);
    std::cout<<"finish nvshmem_free"<<std::endl;
    nvshmem_finalize();
    std::cout<<"finish nvshmem_finalize"<<std::endl;
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout<<"finish destroy the stream"<<std::endl;
    MPI_Finalize();
    std::cout<<"finish MPI_Finalize"<<std::endl;
    return 0;
}
