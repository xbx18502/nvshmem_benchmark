#include "nvshmem.h"
#include "nvshmemx.h"
#include "utils.h"
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#define THREADS_PER_WARP 32
#define THREADS_PER_BLOCK 1024
#define num_blocks 4
#define MAX_MSG_SIZE (1 << 20)

// source: nvshmem/perftest/device/pt-to-pt/shmem_put_bw.cu
__global__ void latency_nbi_block(int *data_d, volatile unsigned int *counter_d,
                                  int len, int pe, int iter) {
  int i, peer;
  unsigned int counter;
  int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z +
             threadIdx.z);
  int bid = blockIdx.x;
  int nblocks = gridDim.x;

  peer = !pe;
  for (i = 0; i < iter; i++) {
    nvshmemx_int_put_nbi_block(data_d + (bid * (len / nblocks)),
                               data_d + (bid * (len / nblocks)), len / nblocks,
                               peer);

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
// not sure, port from block
__global__ void latency_nbi_warp(int *data_d, volatile unsigned int *counter_d,
                                 int len, int pe, int iter) {
  int i, peer;
  unsigned int counter;
  int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x +
            threadIdx.x;
  int bid = blockIdx.x;
  int nblocks = gridDim.x;
  int warp_id = tid / warpSize;
  int warps_per_block =
      (blockDim.x * blockDim.y * blockDim.z + warpSize - 1) / warpSize;
  if (!warps_per_block)
    warps_per_block = 1;
  int block_span = len / nblocks;
  int warp_share = block_span / warps_per_block;
  int warp_remainder = block_span % warps_per_block;
  int warp_offset = warp_share * warp_id +
                    (warp_id < warp_remainder ? warp_id : warp_remainder);
  int warp_elems = warp_share + (warp_id < warp_remainder ? 1 : 0);
  int *warp_ptr = data_d + bid * block_span + warp_offset;
  const bool warp_active = warp_elems > 0;
  peer = !pe;
  for (i = 0; i < iter; i++) {
    if (warp_active) {
      nvshmemx_int_put_nbi_warp(warp_ptr, warp_ptr, warp_elems, peer);
    }

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
// source: nvshmem/perftest/device/pt-to-pt/shmem_put_latency.cu
__global__ void latency_nbi(int *data_d, int len, int pe, int iter) {
  int tid = threadIdx.x;
  int threads = blockDim.x;
  int peer = !pe;
  int quot = threads ? len / threads : 0;
  int rem = threads ? len % threads : 0;
  int base = tid * quot + (tid < rem ? tid : rem);
  int count = quot + (tid < rem ? 1 : 0);

  for (int i = 0; i < iter; i++) {
    if (count > 0) {
      nvshmem_int_put_nbi(data_d + base, data_d + base, count, peer);
    }
    __syncthreads();
    if (!tid) {
      nvshmem_quiet();
    }
    __syncthreads();
  }
}

// source: nvshmem/examples/ring-bcast.cu, but add a loop
__global__ void latency(int *data, size_t nelem, int root, int iter) {
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int peer = (mype + 1) % npes;
  int tid = threadIdx.x;
  int threads = blockDim.x;
  size_t quot = threads ? nelem / threads : 0;
  size_t rem = threads ? nelem % threads : 0;
  size_t base = tid * quot + (tid < rem ? tid : rem);
  size_t count = quot + (tid < rem ? 1 : 0);

  for (int i = 0; i < iter; i++) {
    if (count > 0) {
      nvshmem_int_put(data + base, data + base, count, peer);
    }
    nvshmem_fence();
  }
}
// source: nvshmem/perftest/device/pt-to-pt/shmem_put_latency.cu
#define LATENCY_THREADGROUP(group)                                             \
  __global__ void latency_##group(int *data_d, int len, int pe, int iter) {    \
    int i, tid, peer;                                                          \
                                                                               \
    peer = !pe;                                                                \
    tid = threadIdx.x;                                                         \
                                                                               \
    for (i = 0; i < iter; i++) {                                               \
      nvshmemx_int_put_##group(data_d, data_d, len, peer);                     \
                                                                               \
      __syncthreads();                                                         \
      if (!tid)                                                                \
        nvshmem_quiet();                                                       \
      __syncthreads();                                                         \
    }                                                                          \
  }

LATENCY_THREADGROUP(warp)
LATENCY_THREADGROUP(block)

int main(int c, char *v[]) {
  int mype, npes, size;
  int *data_d = NULL;

  int iter = 200;
  int skip = 20;
  int max_msg_size = MAX_MSG_SIZE;

  int array_size, i;
  void **h_tables;
  uint64_t *h_size_arr;
  double *h_lat;

  float milliseconds;
  cudaEvent_t start, stop;

  init_wrapper(&c, &v);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  mype = nvshmem_my_pe();
  npes = nvshmem_n_pes();

  if (npes != 2) {
    fprintf(stderr, "This test requires exactly two processes \n");
    goto finalize;
  }

  data_d = (int *)nvshmem_malloc(max_msg_size);
  CUDA_CHECK(cudaMemset(data_d, 0, max_msg_size));

  array_size = floor(std::log2((float)max_msg_size)) + 1;
  alloc_tables(&h_tables, 2, array_size);
  h_size_arr = (uint64_t *)h_tables[0];
  h_lat = (double *)h_tables[1];

  nvshmem_barrier_all();

  CUDA_CHECK(cudaDeviceSynchronize());

  i = 0;
  for (size = sizeof(int); size <= max_msg_size; size *= 2) {
    if (!mype) {
      int nelems;
      h_size_arr[i] = size;
      nelems = size / sizeof(int);

      latency<<<num_blocks, THREADS_PER_BLOCK>>>(data_d, nelems, mype, skip);
      cudaEventRecord(start);
      latency<<<num_blocks, THREADS_PER_BLOCK>>>(data_d, nelems, mype, iter);
      cudaEventRecord(stop);

      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventSynchronize(stop));

      /* give latency in us */
      cudaEventElapsedTime(&milliseconds, start, stop);
      h_lat[i] = (milliseconds * 1000) / iter;
      i++;
    }

    nvshmem_barrier_all();
  }

  if (mype == 0) {
    print_table("shmem_put_latency", "Thread", "size (Bytes)", "latency", "us",
                '-', h_size_arr, h_lat, i);
  }

  i = 0;
  for (size = sizeof(int); size <= max_msg_size; size *= 2) {
    if (!mype) {
      int nelems;
      h_size_arr[i] = size;
      nelems = size / sizeof(int);

      latency_warp<<<num_blocks, THREADS_PER_BLOCK>>>(data_d, nelems, mype,
                                                     skip);
      cudaEventRecord(start);
      latency_warp<<<num_blocks, THREADS_PER_BLOCK>>>(data_d, nelems, mype,
                                                     iter);
      cudaEventRecord(stop);

      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventSynchronize(stop));

      /* give latency in us */
      cudaEventElapsedTime(&milliseconds, start, stop);
      h_lat[i] = (milliseconds * 1000) / iter;
      i++;
    }

    nvshmem_barrier_all();
  }

  if (mype == 0) {
    print_table("shmem_put_latency", "Warp", "size (Bytes)", "latency", "us",
                '-', h_size_arr, h_lat, i);
  }

  i = 0;
  for (size = sizeof(int); size <= max_msg_size; size *= 2) {
    if (!mype) {
      int nelems;
      h_size_arr[i] = size;
      nelems = size / sizeof(int);

      latency_block<<<num_blocks, THREADS_PER_BLOCK>>>(data_d, nelems, mype,
                                                       skip);
      cudaEventRecord(start);
      latency_block<<<num_blocks, THREADS_PER_BLOCK>>>(data_d, nelems, mype,
                                                       iter);
      cudaEventRecord(stop);

      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventSynchronize(stop));

      /* give latency in us */
      cudaEventElapsedTime(&milliseconds, start, stop);
      h_lat[i] = (milliseconds * 1000) / iter;
      i++;
    }

    nvshmem_barrier_all();
  }

  if (mype == 0) {
    print_table("shmem_put_latency", "Block", "size (Bytes)", "latency", "us",
                '-', h_size_arr, h_lat, i);
  }

  i = 0;
  for (size = sizeof(int); size <= max_msg_size; size *= 2) {
    if (!mype) {
      int nelems;
      h_size_arr[i] = size;
      nelems = size / sizeof(int);

      latency_nbi<<<num_blocks, THREADS_PER_BLOCK>>>(data_d, nelems, mype,
                                                     skip);
      cudaEventRecord(start);
      latency_nbi<<<num_blocks, THREADS_PER_BLOCK>>>(data_d, nelems, mype,
                                                     iter);
      cudaEventRecord(stop);

      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventSynchronize(stop));

      /* give latency in us */
      cudaEventElapsedTime(&milliseconds, start, stop);
      h_lat[i] = (milliseconds * 1000) / iter;
      i++;
    }

    nvshmem_barrier_all();
  }

  if (mype == 0) {
    print_table("shmem_put_latency", "nbi thread", "size (Bytes)", "latency",
                "us", '-', h_size_arr, h_lat, i);
  }

  i = 0;
  unsigned int *counter_d;
  CUDA_CHECK(cudaMalloc((void **)&counter_d, sizeof(unsigned int) * 2));
  for (size = sizeof(int); size <= max_msg_size; size *= 2) {
    if (!mype) {
      int nelems;
      h_size_arr[i] = size;
      nelems = size / sizeof(int);
      CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
      latency_nbi_warp<<<num_blocks, THREADS_PER_BLOCK>>>(data_d, counter_d,
                                                          nelems, mype, skip);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
      cudaEventRecord(start);
      latency_nbi_warp<<<num_blocks, THREADS_PER_BLOCK>>>(data_d, counter_d,
                                                          nelems, mype, iter);
      cudaEventRecord(stop);

      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventSynchronize(stop));

      /* give latency in us */
      cudaEventElapsedTime(&milliseconds, start, stop);
      h_lat[i] = (milliseconds * 1000) / iter;
      i++;
    }

    nvshmem_barrier_all();
  }

  if (mype == 0) {
    print_table("shmem_put_latency", "nbi warp", "size (Bytes)", "latency",
                "us", '-', h_size_arr, h_lat, i);
  }
  i = 0;
  for (size = sizeof(int); size <= max_msg_size; size *= 2) {
    if (!mype) {
      int nelems;
      h_size_arr[i] = size;
      nelems = size / sizeof(int);
      CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
      latency_nbi_block<<<num_blocks, THREADS_PER_BLOCK>>>(data_d, counter_d,
                                                           nelems, mype, skip);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
      cudaEventRecord(start);
      latency_nbi_block<<<num_blocks, THREADS_PER_BLOCK>>>(data_d, counter_d,
                                                           nelems, mype, iter);
      cudaEventRecord(stop);

      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventSynchronize(stop));

      /* give latency in us */
      cudaEventElapsedTime(&milliseconds, start, stop);
      h_lat[i] = (milliseconds * 1000) / iter;
      i++;
    }

    nvshmem_barrier_all();
  }

  if (mype == 0) {
    print_table("shmem_put_latency", "nbi block", "size (Bytes)", "latency",
                "us", '-', h_size_arr, h_lat, i);
  }
finalize:

  if (data_d)
    nvshmem_free(data_d);
  free_tables(h_tables, 2);

  finalize_wrapper();

  return 0;
}