cmake_minimum_required(VERSION 3.10)
project(sums-of-powers LANGUAGES CUDA)

add_executable(main_n44 main_n44.cu)
set_property(TARGET main_n44 PROPERTY CUDA_ARCHITECTURES native)
