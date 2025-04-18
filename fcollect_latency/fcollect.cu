 #include "coll_test.h"
#include <cstdint>
 #define DATATYPE int64_t
 
 #define CALL_FCOLLECT(TYPENAME, TYPE, TG_PRE, THREADGROUP, THREAD_COMP, ELEM_COMP)                \
     __global__ void test_##TYPENAME##_fcollect_call_kern##THREADGROUP(                            \
         nvshmem_team_t team, TYPE *dest, const TYPE *source, int nelems, int mype, int iter) {    \
         int i;                                                                                    \
                                                                                                   \
         if (!blockIdx.x && (threadIdx.x < THREAD_COMP) && (nelems < ELEM_COMP)) {                 \
             for (i = 0; i < iter; i++) {                                                          \
                 nvshmem##TG_PRE##_##TYPENAME##_fcollect##THREADGROUP(team, dest, source, nelems); \
             }                                                                                     \
         }                                                                                         \
     }
 
 CALL_FCOLLECT(int32, int32_t, , , 1, 512);
 CALL_FCOLLECT(int64, int64_t, , , 1, 512);
 CALL_FCOLLECT(int32, int32_t, x, _warp, warpSize, 4096);
 CALL_FCOLLECT(int64, int64_t, x, _warp, warpSize, 4096);
 CALL_FCOLLECT(int32, int32_t, x, _block, INT_MAX, INT_MAX);
 CALL_FCOLLECT(int64, int64_t, x, _block, INT_MAX, INT_MAX);
 
 int fcollect_calling_kernel(nvshmem_team_t team, void *dest, const void *source, int mype,
                             int max_elems, cudaStream_t stream, void **h_tables) {
     int status = 0;
     int nvshm_test_num_tpb = TEST_NUM_TPB_BLOCK;
     int num_blocks = 1;
     int num_elems = 1;
     int i;
     int skip = MAX_SKIP;
     int iter = MAX_ITERS;
     uint64_t *h_size_array = (uint64_t *)h_tables[0];
     double *h_thread_lat = (double *)h_tables[1];
     double *h_warp_lat = (double *)h_tables[2];
     double *h_block_lat = (double *)h_tables[3];
     float milliseconds;
     void *args_1[] = {&team, &dest, &source, &num_elems, &mype, &skip};
     void *args_2[] = {&team, &dest, &source, &num_elems, &mype, &iter};
     cudaEvent_t start, stop;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);

    int max_msg_size = 1048576;
 
     nvshmem_barrier_all();
     i = 0;
     for (num_elems = 1; num_elems < max_msg_size/sizeof(int32_t); num_elems *= 2) {
         status = nvshmemx_collective_launch((const void *)test_int32_fcollect_call_kern, num_blocks,
                                             nvshm_test_num_tpb, args_1, 0, stream);
         if (status != NVSHMEMX_SUCCESS) {
             fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
             exit(-1);
         }
         CUDA_CHECK(cudaStreamSynchronize(stream));
 
         nvshmem_barrier_all();
 
         cudaEventRecord(start, stream);
         status = nvshmemx_collective_launch((const void *)test_int32_fcollect_call_kern, num_blocks,
                                             nvshm_test_num_tpb, args_2, 0, stream);
        // test_int32_fcollect_call_kern<<< num_blocks, 
        // nvshm_test_num_tpb,0>>>(team, (int32_t*)dest
        // ,(int32_t*)source,num_elems, mype,iter);

         if (status != NVSHMEMX_SUCCESS) {
             fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
             exit(-1);
         }
         cudaEventRecord(stop, stream);
         CUDA_CHECK(cudaStreamSynchronize(stream));
 
         if (!mype) {
             cudaEventElapsedTime(&milliseconds, start, stop);
             h_thread_lat[i] = (milliseconds * 1000.0) / (float)iter;
         }
         i++;
         nvshmem_barrier_all();
     }
 
     i = 0;
     for (num_elems = 1; num_elems < max_msg_size/sizeof(int32_t); num_elems *= 2) {
         status = nvshmemx_collective_launch((const void *)test_int32_fcollect_call_kern_warp,
                                             num_blocks, nvshm_test_num_tpb, args_1, 0, stream);
         if (status != NVSHMEMX_SUCCESS) {
             fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
             exit(-1);
         }
         CUDA_CHECK(cudaStreamSynchronize(stream));
 
         nvshmem_barrier_all();
 
         cudaEventRecord(start, stream);
         status = nvshmemx_collective_launch((const void *)test_int32_fcollect_call_kern_warp,
                                             num_blocks, nvshm_test_num_tpb, args_2, 0, stream);
         if (status != NVSHMEMX_SUCCESS) {
             fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
             exit(-1);
         }
         cudaEventRecord(stop, stream);
         CUDA_CHECK(cudaStreamSynchronize(stream));
 
         if (!mype) {
             cudaEventElapsedTime(&milliseconds, start, stop);
             h_warp_lat[i] = (milliseconds * 1000.0) / (float)iter;
         }
         i++;
         nvshmem_barrier_all();
     }
 
     i = 0;
     for (num_elems = 1; num_elems < max_msg_size/sizeof(int32_t); num_elems *= 2) {
         h_size_array[i] = num_elems * 4;
         status = nvshmemx_collective_launch((const void *)test_int32_fcollect_call_kern_block,
                                             num_blocks, nvshm_test_num_tpb, args_1, 0, stream);
         if (status != NVSHMEMX_SUCCESS) {
             fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
             exit(-1);
         }
         CUDA_CHECK(cudaStreamSynchronize(stream));
 
         nvshmem_barrier_all();
 
         cudaEventRecord(start, stream);
         status = nvshmemx_collective_launch((const void *)test_int32_fcollect_call_kern_block,
                                             num_blocks, nvshm_test_num_tpb, args_2, 0, stream);
         if (status != NVSHMEMX_SUCCESS) {
             fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
             exit(-1);
         }
         cudaEventRecord(stop, stream);
         CUDA_CHECK(cudaStreamSynchronize(stream));
 
         if (!mype) {
             cudaEventElapsedTime(&milliseconds, start, stop);
             h_block_lat[i] = (milliseconds * 1000.0) / (float)iter;
         }
         i++;
         nvshmem_barrier_all();
     }
 
     if (!mype) {
         print_table("fcollect_device", "32-bit-thread", "size (Bytes)", "latency", "us", '-',
                     h_size_array, h_thread_lat, i);
         print_table("fcollect_device", "32-bit-warp", "size (Bytes)", "latency", "us", '-',
                     h_size_array, h_warp_lat, i);
         print_table("fcollect_device", "32-bit-block", "size (Bytes)", "latency", "us", '-',
                     h_size_array, h_block_lat, i);
     }
 
     i = 0;
     for (num_elems = 1; num_elems < max_msg_size/sizeof(int64_t); num_elems *= 2) {
         status = nvshmemx_collective_launch((const void *)test_int64_fcollect_call_kern, num_blocks,
                                             nvshm_test_num_tpb, args_1, 0, stream);
         if (status != NVSHMEMX_SUCCESS) {
             fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
             exit(-1);
         }
         CUDA_CHECK(cudaStreamSynchronize(stream));
 
         nvshmem_barrier_all();
 
         cudaEventRecord(start, stream);
         status = nvshmemx_collective_launch((const void *)test_int64_fcollect_call_kern, num_blocks,
                                             nvshm_test_num_tpb, args_2, 0, stream);
         if (status != NVSHMEMX_SUCCESS) {
             fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
             exit(-1);
         }
         cudaEventRecord(stop, stream);
         CUDA_CHECK(cudaStreamSynchronize(stream));
 
         if (!mype) {
             cudaEventElapsedTime(&milliseconds, start, stop);
             h_thread_lat[i] = (milliseconds * 1000.0) / (float)iter;
         }
         i++;
         nvshmem_barrier_all();
     }
 
     i = 0;
     for (num_elems = 1; num_elems < max_msg_size/sizeof(int64_t); num_elems *= 2) {
         status = nvshmemx_collective_launch((const void *)test_int64_fcollect_call_kern_warp,
                                             num_blocks, nvshm_test_num_tpb, args_1, 0, stream);
         if (status != NVSHMEMX_SUCCESS) {
             fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
             exit(-1);
         }
         CUDA_CHECK(cudaStreamSynchronize(stream));
 
         nvshmem_barrier_all();
 
         cudaEventRecord(start, stream);
         status = nvshmemx_collective_launch((const void *)test_int64_fcollect_call_kern_warp,
                                             num_blocks, nvshm_test_num_tpb, args_2, 0, stream);
         if (status != NVSHMEMX_SUCCESS) {
             fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
             exit(-1);
         }
         cudaEventRecord(stop, stream);
         CUDA_CHECK(cudaStreamSynchronize(stream));
 
         if (!mype) {
             cudaEventElapsedTime(&milliseconds, start, stop);
             h_warp_lat[i] = (milliseconds * 1000.0) / (float)iter;
         }
         i++;
         nvshmem_barrier_all();
     }
 
     i = 0;
     for (num_elems = 1; num_elems < max_msg_size/sizeof(int64_t); num_elems *= 2) {
         h_size_array[i] = num_elems * 8;
         status = nvshmemx_collective_launch((const void *)test_int64_fcollect_call_kern_block,
                                             num_blocks, nvshm_test_num_tpb, args_1, 0, stream);
         if (status != NVSHMEMX_SUCCESS) {
             fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
             exit(-1);
         }
         CUDA_CHECK(cudaStreamSynchronize(stream));
 
         nvshmem_barrier_all();
 
         cudaEventRecord(start, stream);
         status = nvshmemx_collective_launch((const void *)test_int64_fcollect_call_kern_block,
                                             num_blocks, nvshm_test_num_tpb, args_2, 0, stream);
         if (status != NVSHMEMX_SUCCESS) {
             fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
             exit(-1);
         }
         cudaEventRecord(stop, stream);
         CUDA_CHECK(cudaStreamSynchronize(stream));
 
         if (!mype) {
             cudaEventElapsedTime(&milliseconds, start, stop);
             h_block_lat[i] = (milliseconds * 1000.0) / (float)iter;
         }
         i++;
         nvshmem_barrier_all();
     }
 
     if (!mype) {
         print_table("fcollect_device", "64-bit-thread", "size (Bytes)", "latency", "us", '-',
                     h_size_array, h_thread_lat, i);
         print_table("fcollect_device", "64-bit-warp", "size (Bytes)", "latency", "us", '-',
                     h_size_array, h_warp_lat, i);
         print_table("fcollect_device", "64-bit-block", "size (Bytes)", "latency", "us", '-',
                     h_size_array, h_block_lat, i);
     }
 
     return status;
 }
 
 int main(int argc, char **argv) {
     int status = 0;
     int mype, npes, array_size, max_elems;
     char *value = NULL;
     // size needs to hold psync array, source array (nelems) and dest array (nelems * npes)
     size_t size = (MAX_ELEMS * (MAX_NPES + 1)) * sizeof(DATATYPE);
     size_t alloc_size;
     int num_elems;
     DATATYPE *h_buffer = NULL;
     DATATYPE *d_buffer = NULL;
     DATATYPE *d_source, *d_dest;
     DATATYPE *h_source, *h_dest;
     char size_string[100];
     cudaStream_t cstrm;
     void **h_tables;
 
     max_elems = (MAX_ELEMS / 2);
 
     if (NULL != value) {
         max_elems = atoi(value);
         if (0 == max_elems) {
             fprintf(stderr, "Warning: min max elem size = 1\n");
             max_elems = 1;
         }
     }
 
     array_size = floor(std::log2((float)max_elems)) + 1;
 
     DEBUG_PRINT("symmetric size %lu\n", size);
     sprintf(size_string, "%lu", size);
 
     status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
     if (status) {
         fprintf(stderr, "setenv failed \n");
         status = -1;
         goto out;
     }
 
     init_wrapper(&argc, &argv);
     alloc_tables(&h_tables, 4, array_size);
 
     mype = nvshmem_my_pe();
     npes = nvshmem_n_pes();
     assert(npes <= MAX_NPES);
     CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));
 
     num_elems = MAX_ELEMS / 2;
     alloc_size = (num_elems * (MAX_NPES + 1)) * sizeof(DATATYPE);
 
     CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
     h_source = (DATATYPE *)h_buffer;
     h_dest = (DATATYPE *)&h_source[num_elems];
 
     d_buffer = (DATATYPE *)nvshmem_malloc(alloc_size);
     if (!d_buffer) {
         fprintf(stderr, "nvshmem_malloc failed \n");
         status = -1;
         goto out;
     }
 
     d_source = (DATATYPE *)d_buffer;
     d_dest = (DATATYPE *)&d_source[num_elems];
 
     CUDA_CHECK(cudaMemcpyAsync(d_source, h_source, (sizeof(DATATYPE) * num_elems),
                                cudaMemcpyHostToDevice, cstrm));
     CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest, (sizeof(DATATYPE) * num_elems * npes),
                                cudaMemcpyHostToDevice, cstrm));
 
     fcollect_calling_kernel(NVSHMEM_TEAM_WORLD, (void *)d_dest, (void *)d_source, mype, max_elems,
                             cstrm, h_tables);
 
     CUDA_CHECK(cudaMemcpyAsync(h_source, d_source, (sizeof(DATATYPE) * num_elems),
                                cudaMemcpyDeviceToHost, cstrm));
     CUDA_CHECK(cudaMemcpyAsync(h_dest, d_dest, (sizeof(DATATYPE) * num_elems * npes),
                                cudaMemcpyDeviceToHost, cstrm));
 
     nvshmem_barrier_all();
 
     CUDA_CHECK(cudaFreeHost(h_buffer));
     nvshmem_free(d_buffer);
 
     CUDA_CHECK(cudaStreamDestroy(cstrm));
     free_tables(h_tables, 4);
     finalize_wrapper();
 
 out:
     return 0;
 }
 