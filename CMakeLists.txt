cmake_minimum_required(VERSION 3.23.1)
project(iPreGraph LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Honor an explicitly selected configuration; optimize single-config builds by default.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

include(CTest)
find_package(Threads REQUIRED)

option(IPREGRAPH_NATIVE_ARCH "Optimize for the build machine's CPU" OFF)

# Keep compiler options on the project targets instead of replacing user flags.
add_library(ipregraph_options INTERFACE)
target_include_directories(ipregraph_options INTERFACE "${PROJECT_SOURCE_DIR}/include")
target_compile_features(ipregraph_options INTERFACE cxx_std_17)
target_link_libraries(ipregraph_options INTERFACE Threads::Threads)

if(MSVC)
    target_compile_options(ipregraph_options INTERFACE /W3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ipregraph_options INTERFACE -Wall -Wextra)
endif()

if(IPREGRAPH_NATIVE_ARCH)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" IPREGRAPH_HAS_MARCH_NATIVE)
    if(IPREGRAPH_HAS_MARCH_NATIVE)
        target_compile_options(ipregraph_options INTERFACE -march=native)
    else()
        message(FATAL_ERROR "IPREGRAPH_NATIVE_ARCH requires a compiler supporting -march=native")
    endif()
endif()

add_subdirectory(tests)
