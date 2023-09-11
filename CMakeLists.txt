cmake_minimum_required(VERSION 3.25.2)
project(sums-of-powers LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# import third-party libraries:
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)

set(fmt_patch git apply ${CMAKE_CURRENT_SOURCE_DIR}/fmt-fix-nvcc-and-int128.patch)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG 10.1.1
  PATCH_COMMAND ${fmt_patch}
  UPDATE_DISCONNECTED 1
)

FetchContent_MakeAvailable(googletest fmt)

# actual targets
add_executable(
  main_n44 
  main_n44.cu
)
set_property(
  TARGET 
  main_n44 
  PROPERTY CUDA_ARCHITECTURES native
)
target_link_libraries(
  main_n44
  PRIVATE fmt::fmt
)

# testing with googletest
enable_testing()
add_executable(basic_test basic_test.cu)
target_link_libraries(
  basic_test
  GTest::gtest_main
  fmt::fmt
)
include(GoogleTest)
gtest_discover_tests(basic_test)
