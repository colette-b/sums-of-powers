cmake_minimum_required(VERSION 3.25.2)
project(sums-of-powers LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# use googletest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
FetchContent_MakeAvailable(googletest)

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

# testing
enable_testing()
add_executable(basic_test basic_test.cu)
target_link_libraries(
  basic_test
  GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(basic_test)
