# cmake_minimum_required(VERSION 3.20)

# project(roofline LANGUAGES CXX)

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(OpenMP REQUIRED)

# add_executable(roofline
#     roofline_gstream.cpp
# )

# target_compile_options(roofline PRIVATE
#     -O3
#     -march=native
#     -Wall
#     -Wextra
#     -Wpedantic
#     -fopenmp
#     -fopenmp-targets=nvptx64-nvidia-cuda
#     --offload-arch=sm_120
#     --cuda-path=/usr/local/cuda-12.8
# )

# target_link_options(roofline PRIVATE
#     -fopenmp
#     -fopenmp-targets=nvptx64-nvidia-cuda
#     --offload-arch=sm_120
#     --cuda-path=/usr/local/cuda-12.8
# )
# target_link_libraries(roofline PRIVATE OpenMP::OpenMP_CXX)


cmake_minimum_required(VERSION 3.20)
project(roofline_benchmark CXX)

add_executable(roofline roofline_stream.cpp)

# Enable standard OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(roofline PRIVATE OpenMP::OpenMP_CXX)

# Clang specific offloading flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(roofline PRIVATE 
        -fopenmp
        -fopenmp-targets=nvptx64-nvidia-cuda
        --cuda-path=/usr/local/cuda-12.8
        --libomptarget-nvptx-bc-path=/usr/local/cuda-12.8/nvvm/libdevice/libdevice.10.bc
    )
    target_link_options(roofline PRIVATE 
        -fopenmp              # <--- Added this missing flag here
        -fopenmp-targets=nvptx64-nvidia-cuda
        --cuda-path=/usr/local/cuda-12.8
    )
endif()

