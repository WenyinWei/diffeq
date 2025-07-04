cmake_minimum_required(VERSION 3.20)
project(diffeq 
    VERSION 1.0.0
    DESCRIPTION "Modern C++ ODE Integration Library with Async Signal Processing"
    LANGUAGES CXX)

# Set C++20 as the standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE MATCHES "Release")
        add_compile_options(-O3 -DNDEBUG)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4)
    if(CMAKE_BUILD_TYPE MATCHES "Release")
        add_compile_options(/O2 /DNDEBUG)
    endif()
endif()

# Header-only library interface
add_library(diffeq INTERFACE)
target_include_directories(diffeq INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Require C++20
target_compile_features(diffeq INTERFACE cxx_std_20)

# Optional dependencies for enhanced features
option(DIFFEQ_ENABLE_TESTS "Build tests" ON)
option(DIFFEQ_ENABLE_EXAMPLES "Build examples" ON)
option(DIFFEQ_ENABLE_BENCHMARKS "Build benchmarks" OFF)

# Find threading support (required for async operations)
find_package(Threads REQUIRED)
target_link_libraries(diffeq INTERFACE Threads::Threads)

# Optional: std::execution support detection
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
    #include <execution>
    int main() { 
        return std::execution::par != std::execution::seq; 
    }" DIFFEQ_HAS_STD_EXECUTION)

if(DIFFEQ_HAS_STD_EXECUTION)
    target_compile_definitions(diffeq INTERFACE DIFFEQ_HAS_STD_EXECUTION=1)
else()
    target_compile_definitions(diffeq INTERFACE DIFFEQ_HAS_STD_EXECUTION=0)
endif()

# Examples
if(DIFFEQ_ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(DIFFEQ_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Benchmarks
if(DIFFEQ_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS diffeq
    EXPORT diffeqTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT diffeqTargets
    FILE diffeqTargets.cmake
    NAMESPACE diffeq::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/diffeq
)

# Generate package config
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/diffeqConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/diffeqConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/diffeqConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/diffeq
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/diffeqConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/diffeqConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/diffeq
)
