cmake_minimum_required(VERSION 3.26)
project(treeson)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(treeson example.cpp)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

add_executable(multitarg_example example_multitarg.cpp)  # Replace main.cpp with your source file

target_link_libraries(multitarg_example arrow parquet)

