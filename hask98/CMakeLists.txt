cmake_minimum_required(VERSION 3.4.3)
project(hask98 C CXX)

set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -O0 -fsanitize=address,undefined -static-libasan")
set(CMAKE_CXX_FLAGS "-O0 -Wall -Werror -Wno-error=unused-result")
add_executable(hask98 main.cpp)
set_target_properties(hask98 PROPERTIES CXX_STANDARD 14)
