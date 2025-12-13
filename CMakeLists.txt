cmake_minimum_required(VERSION 3.14)
project(llvc_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(LLVC_SOURCES
    src/tensor.cpp
    src/audio.cpp
    src/weights.cpp
    src/activations.cpp
    src/layers.cpp
    src/attention.cpp
    src/blocks.cpp
    src/model.cpp
)

# Library
add_library(llvc STATIC ${LLVC_SOURCES})
target_include_directories(llvc PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Main executable
add_executable(llvc_infer src/main.cpp)
target_link_libraries(llvc_infer PRIVATE llvc)

# Test executable (optional)
option(BUILD_TESTS "Build test executables" OFF)
if(BUILD_TESTS)
    add_executable(llvc_test test/test_main.cpp)
    target_link_libraries(llvc_test PRIVATE llvc)
endif()

# Installation
install(TARGETS llvc_infer DESTINATION bin)
install(TARGETS llvc DESTINATION lib)
install(DIRECTORY include/llvc DESTINATION include)
