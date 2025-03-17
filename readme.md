# NVSHMEM Benchmarks

This project contains benchmarks for NVSHMEM (NVIDIA Symmetric Hierarchical Memory), a parallel programming interface for NVIDIA GPUs. The benchmarks evaluate the performance of NVSHMEM operations such as put bandwidth, put latency, and collective operations.

## Prerequisites

- NVIDIA GPU with CUDA support
- NVSHMEM library
- MPI implementation (such as OpenMPI or MPICH)
- CMake (version 3.10 or newer)
- C/C++ compiler with CUDA support

## Building the Benchmarks

Follow these steps to build the benchmarks:

```bash
mkdir -p build
cd build
cmake ..
make -j
make install
```

## Available Benchmarks

The repository includes several benchmarks:

- **put_bw**: Measures the bandwidth of NVSHMEM put operations
- **put_latency**: Measures the latency of NVSHMEM put operations
- **put_latency_nbi**: Measures the latency of non-blocking NVSHMEM put operations
- **fcollect_latency**: Measures the latency of NVSHMEM fcollect collective operations
- **mpi_init_put_bw**: Measures put bandwidth with MPI initialization

## Running the Benchmarks

Each benchmark directory contains a shell script for running the benchmark. For example:

```bash
# To run the NVSHMEM put bandwidth benchmark
./osu_nvshmem_put_bw.sh

# To run the SHMEM put bandwidth benchmark
./shmem_put_bw.sh
```

## Benchmark Options

The benchmarks support various options. For example, the put bandwidth benchmark may accept parameters such as:

- Message size
- Number of iterations
- Warmup iterations
- Number of processes

For specific options, please refer to the source code or run the benchmark with the `--help` flag.

## Project Structure

- **attribute_based.cu**: Implementation of attribute-based NVSHMEM operations
- **osu_nvshmem_put_bw.cu**: NVSHMEM put bandwidth benchmark
- **shmem_put_bw.cu**: Standard SHMEM put bandwidth benchmark
- **fcollect_latency/**: Directory containing fcollect latency benchmark
- **put_latency/**: Directory containing put latency benchmark
- **put_latency_nbi/**: Directory containing non-blocking put latency benchmark

## License

Please refer to the project's license file for licensing information.