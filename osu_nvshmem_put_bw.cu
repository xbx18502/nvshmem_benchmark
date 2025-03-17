#define BENCHMARK "OSU NVSHMEM Put Bandwidth Test"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "osu_util_pgas.h"

#define MAX_MSG_SIZE (32 * 1024 * 1024)
#define SKIP 1000
#define LOOP 10000
#define SKIP_LARGE 10  
#define LOOP_LARGE 100
#define LARGE_MESSAGE_SIZE 8192

int main(int argc, char *argv[]) {
    int mype, npes;
    double *s_buf = NULL, *r_buf = NULL;
    int size, i;
    double t_start = 0.0, t_end = 0.0;
    double mb_total = 0.0, t_total = 0.0;
    
    // Initialize NVSHMEM
    nvshmem_init();
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    
    if (npes != 2) {
        if (mype == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }
        nvshmem_finalize();
        return EXIT_FAILURE;
    }

    // Allocate CUDA memory using NVSHMEM
    s_buf = (double *)nvshmem_malloc(MAX_MSG_SIZE);
    r_buf = (double *)nvshmem_malloc(MAX_MSG_SIZE);
    
    if (mype == 0) {
        fprintf(stdout, HEADER);
        fprintf(stdout, "%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Bandwidth (MB/s)");
        fflush(stdout);
    }

    for (size = 1; size <= MAX_MSG_SIZE; size = (size ? size * 2 : 1)) {
        int loop = (size > LARGE_MESSAGE_SIZE) ? LOOP_LARGE : LOOP;
        int skip = (size > LARGE_MESSAGE_SIZE) ? SKIP_LARGE : SKIP;

        nvshmem_barrier_all();

        if (mype == 0) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            // Warmup
            for (i = 0; i < skip; i++) {
                nvshmem_putmem(r_buf, s_buf, size, 1);
            }
            
            cudaEventRecord(start);
            for (i = 0; i < loop; i++) {
                nvshmem_putmem(r_buf, s_buf, size, 1);
            }
            cudaEventRecord(stop);
            
            float milliseconds = 0.0f;
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            
            mb_total = size * loop / (1.0 * 1e6);
            t_total = milliseconds / 1e3;
            double bw = mb_total / t_total;
            
            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, bw);
            fflush(stdout);
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }

    nvshmem_barrier_all();
    
    if (s_buf) nvshmem_free(s_buf);
    if (r_buf) nvshmem_free(r_buf);
    
    nvshmem_finalize();
    return EXIT_SUCCESS;
}