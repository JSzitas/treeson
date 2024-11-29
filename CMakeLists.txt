cmake_minimum_required(VERSION 3.26)
project(treeson)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(treeson examples/example.cpp)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

add_executable(multitarg_example examples/example_multitarg.cpp)
add_executable(profile examples/example_multitarg.cpp)

target_link_libraries(multitarg_example arrow parquet)

