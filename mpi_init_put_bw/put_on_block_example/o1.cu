#include <stdio.h>
#include <assert.h>
#include "nvshmem.h"
#include "nvshmemx.h"

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
    } while (0)

#define THREADS_PER_BLOCK 1024

__global__ void set_and_shift_kernel(float *send_data, float *recv_data, int num_elems, int mype,
                                     int npes) {
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    /* set the corresponding element of send_data */
    if (thread_idx < num_elems) send_data[thread_idx] = mype;

    int peer = (mype + 1) % npes;
    /* Every thread in block 0 calls nvshmemx_float_put_block. Alternatively,
       every thread can call shmem_float_p, but shmem_float_p has a disadvantage
       that when the destination GPU is connected via IB, there will be one rma
       message for every single element which can be detrimental to performance.
       And the disadvantage with shmem_float_put is that when the destination GPU is p2p
       connected, it cannot leverage multiple threads to copy the data to the destination
       GPU. */
    int block_offset = blockIdx.x * blockDim.x;
    nvshmemx_float_put_block(recv_data + block_offset, send_data + block_offset,
                             min(blockDim.x, num_elems - block_offset),
                             peer); /* All threads in a block call the API
                                       with the same arguments */
}

int main(int c, char *v[]) {
    int mype, npes, mype_node;
    float *send_data, *recv_data;
    int num_elems = 8192;
    int num_blocks;

    nvshmem_init();

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    // application picks the device each PE will use
    CUDA_CHECK(cudaSetDevice(mype_node));
    send_data = (float *)nvshmem_malloc(sizeof(float) * num_elems);
    recv_data = (float *)nvshmem_malloc(sizeof(float) * num_elems);
    assert(send_data != NULL && recv_data != NULL);

    assert(num_elems % THREADS_PER_BLOCK == 0); /* for simplicity */
    num_blocks = num_elems / THREADS_PER_BLOCK;

    set_and_shift_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(send_data, recv_data, num_elems, mype,
                                                            npes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Do data validation */
    float *host = new float[num_elems];
    CUDA_CHECK(cudaMemcpy(host, recv_data, num_elems * sizeof(float), cudaMemcpyDefault));
    int ref = (mype - 1 + npes) % npes;
    bool success = true;
    for (int i = 0; i < num_elems; ++i) {
        if (host[i] != ref) {
            printf("Error at %d of rank %d: %f\n", i, mype, host[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("[%d of %d] run complete \n", mype, npes);
    } else {
        printf("[%d of %d] run failure \n", mype, npes);
    }

    nvshmem_free(send_data);
    nvshmem_free(recv_data);

    nvshmem_finalize();

    return 0;
}