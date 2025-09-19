# 设置 CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(nvshmem_benchmark)

# 添加子目录
add_subdirectory(fcollect_latency)
add_subdirectory(put_latency)
add_subdirectory(put_latency_nbi)
add_subdirectory(put_bw)
add_subdirectory(broadcast_host)