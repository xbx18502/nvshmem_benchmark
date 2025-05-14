/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#include "coll_test.h"
#include "host/nvshmem_coll_api.h"
#include <cstdio>
#define DATATYPE int64_t

int main(int argc, char **argv) {
    int status = 0;
    int mype, npes;
    size_t size = MAX_ELEMS * 2 * sizeof(DATATYPE);
    size_t alloc_size;
    int num_elems;
    DATATYPE *buffer = NULL;
    DATATYPE *h_buffer = NULL;
    DATATYPE *d_source, *d_dest;
    DATATYPE *h_source, *h_dest;
    char size_string[100];
    uint64_t size_array[MAX_ELEMS_LOG + 1];
    double latency_array[MAX_ELEMS_LOG + 1];
    cudaStream_t stream;
    int PE_root = 0;

    memset(size_array, 0, (MAX_ELEMS_LOG + 1) * sizeof(uint64_t));
    memset(latency_array, 0, (MAX_ELEMS_LOG + 1) * sizeof(double));

    DEBUG_PRINT("symmetric size requested %lu\n", size);
    sprintf(size_string, "%lu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failed \n");
        status = -1;
        goto out;
    }

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    (void)npes;  // Silence unused variable warning
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // num_elems = MAX_ELEMS / 2;
    num_elems = 1;
    alloc_size = num_elems * 2 * sizeof(DATATYPE);

    CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
    h_source = (DATATYPE *)h_buffer;
    h_dest = (DATATYPE *)&h_source[num_elems];

    buffer = (DATATYPE *)nvshmem_malloc(alloc_size);
    if (!buffer) {
        fprintf(stderr, "nvshmem_malloc failed \n");
        status = -1;
        goto out;
    }
    d_source = (DATATYPE *)buffer;
    d_dest = (DATATYPE *)&d_source[num_elems];
    *h_buffer = 111;
    CUDA_CHECK(cudaMemcpyAsync(d_source, h_source, 1, cudaMemcpyHostToDevice, stream));                      
    CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest, 1,cudaMemcpyHostToDevice, stream));
    nvshmem_int64_broadcast(NVSHMEM_TEAM_WORLD, d_dest, d_source, (size_t)1, PE_root);
    // if (!mype) {
    //     print_table("broadcast", "32-bit", "size (bytes)", "latency", "us", '-', size_array,
    //                 latency_array, MAX_ELEMS_LOG + 1);
    // }
    CUDA_CHECK(cudaMemcpyAsync(h_source, d_source, 1,cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_dest,d_dest, 1,cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // RUN_COLL(broadcast, BCAST, int64, int64_t, d_source, h_source, d_dest, h_dest, npes, PE_root,
    //          stream, size_array, latency_array);
    // if (!mype) {
    //     print_table("broadcast", "64-bit", "size (bytes)", "latency", "us", '-', size_array,
    //                 latency_array, MAX_ELEMS_LOG + 1);
    // }
    nvshmem_barrier_all();  
    printf("h_dest = %ld\n", *h_dest);
    CUDA_CHECK(cudaFreeHost(h_buffer));
    nvshmem_free(buffer);

    nvshmem_barrier_all();

    CUDA_CHECK(cudaStreamDestroy(stream));

    finalize_wrapper();

out:
    return 0;
}
