#include <cstddef>
#include <stdio.h>
#include <iostream>
#include "mpi.h"
#include "nvshmem.h"
#include "nvshmemx.h"
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

int skip = 1;
int loop = 1;
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

__global__ void bw(double* dest, int size, int peer) {
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);
    int bid = blockIdx.x;
    int nblocks = gridDim.x;
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    nvshmemx_double_put_nbi_block(dest + (bid * (size / nblocks)),
                                  dest + (bid * (size / nblocks)), size / nblocks, peer);

}
/** only <<<1,1>>> */
__global__ void bw3(double* dest, int size, int peer, int iter) {
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);
    int bid = blockIdx.x;
    int nblocks = gridDim.x;
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    for(int i=0;i<iter;i++){
        nvshmem_double_put_nbi(dest ,dest , size, peer);
        nvshmem_quiet();
    }
}
/** <<<4,1024>>> */
// ...existing code...
/** replaced bw3: 支持 <<<4,1024>>> 并在 kernel 内做跨-block 同步，最后由最后一个到达的 block 调用 nvshmem_quiet() */
__global__ void bw4(double *data_d, volatile unsigned int *counter_d, int len, int pe, int iter) {
    int i, peer;
    unsigned int counter;
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);
    int bid = blockIdx.x;
    int nblocks = gridDim.x;
    int nthreads = blockDim.x * blockDim.y * blockDim.z;
    peer = !pe;

    int chunk = len / nthreads;
    double *src = data_d + (tid * chunk);
    double *dst = data_d + (tid * chunk);

    for (i = 0; i < iter; i++) {
        // 每个 block 发自己的 chunk，非阻塞
        nvshmem_double_put_nbi(dst, src, chunk, peer);

        // block 内线程先同步
        __syncthreads();

        // block 0 作为本轮到达计数器的代表执行全局原子和等待
        if (!tid) {
            __threadfence(); // 保证先前内存操作可见
            counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
            // 如果是本轮最后一个到达的 block，设置完成标志 (counter_d[1])
            if (counter == (unsigned int)(nblocks * (i + 1) - 1)) {
                *(counter_d + 1) += 1;
            }
            // 等待完成标志被最后一个到达的 block 更新为 i+1
            while (*(counter_d + 1) != (unsigned int)(i + 1))
                ;
        }

        // 等待所有线程看到完成标志后继续下一轮
        __syncthreads();
    }

    // 循环外再做一次同步并让最后一个到达者执行 nvshmem_quiet()
    __syncthreads();
    if (!tid) {
        __threadfence();
        counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
        if (counter == (unsigned int)(nblocks * (i + 1) - 1)) {
            nvshmem_quiet();
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != (unsigned int)(i + 1))
            ;
    }
    __syncthreads();
}
__global__ void bw2(double *data_d, volatile unsigned int *counter_d, int len, int pe, int iter) {
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
    int global_pe = nvshmem_my_pe();
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

    for (size = 1; size <= MAX_MSG_SIZE_PT2PT; size = (size ? size * 2 : 1)) {
        if (size > large_message_size) {
            loop = loop_large = 100;
            skip = skip_large = 0;
        }
        //nvshmemx_barrier_all_on_stream(stream);
        //nvshmem_barrier_all();
        if (0 == mype_node) {
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEventRecord(start);
            for(int i = 0;i<loop;i++){
                bw<<<4,1024>>> (destination, size, mype_node);
                cudaDeviceSynchronize();
                nvshmem_quiet();
            }
            // Wait for kernel to complete on device
            CUDA_CHECK(cudaDeviceSynchronize());
            // Ensure all NVSHMEM puts are completed remotely
            nvshmem_quiet();
            // Record end after quiet so timing includes remote completion
            cudaEventRecord(stop);
            CUDA_CHECK(cudaEventSynchronize(stop));
            nvshmem_barrier_all();
        }
        else{
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
            t_total = milliseconds/1e3;
            double bw = mb_total / t_total;
            fprintf(stdout, "%-*d%*.*f\n", 10, size*8, FIELD_WIDTH,
                    FLOAT_PRECISION, bw);
            fflush(stdout);
            //std::cout<<"PE0 finish print"<<std::endl;
        }
        else{
            std::cout<<"PE1 finish print"<<std::endl;
        }
    }
    cudaDeviceSynchronize();
    nvshmem_barrier_all();
    std::cout<<"naive finished"<<std::endl;
    unsigned int *counter_d;
    // allocate device memory for the counter used by kernels
    CUDA_CHECK(cudaMalloc((void**)&counter_d, sizeof(unsigned int) * 2));
    // Note: individual loops still call cudaMemset(counter_d, 0, ...) before use
    for (size = 1; size <= MAX_MSG_SIZE_PT2PT; size = (size ? size * 2 : 1)) {
        if (size > large_message_size) {
            loop = loop_large = 100;
            skip = skip_large = 0;
        }
        //nvshmemx_barrier_all_on_stream(stream);
        //nvshmem_barrier_all();
        if (0 == mype_node) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            cudaEventRecord(start);

            bw4<<<4,1024>>> (destination, counter_d, size, mype_node, loop);
            // Wait for kernel to complete on device
            CUDA_CHECK(cudaDeviceSynchronize());
            // Ensure all NVSHMEM puts are completed remotely
            nvshmem_quiet();
            // Record end after quiet so timing includes remote completion
            cudaEventRecord(stop);
            CUDA_CHECK(cudaEventSynchronize(stop));
            nvshmem_barrier_all();
        }
        else{
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
            t_total = milliseconds/1e3;
            double bw = mb_total / t_total;
            fprintf(stdout, "%-*d%*.*f\n", 10, size*8, FIELD_WIDTH,
                    FLOAT_PRECISION, bw);
            fflush(stdout);
            //std::cout<<"PE0 finish print"<<std::endl;
        }
        else{
            std::cout<<"PE1 finish print"<<std::endl;
        }
    }
    // free the device counter when no longer needed
    CUDA_CHECK(cudaFree(counter_d));
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

