# Performance Guide

This guide provides comprehensive information about optimizing performance when using the DiffEq library.

## 🚀 Performance Overview

The DiffEq library is designed for high-performance numerical integration with several key optimization strategies:

- **Header-only design** - No runtime linking overhead
- **Template-based** - Compile-time optimization and inlining
- **Modern C++20** - Leverages latest compiler optimizations
- **Parallel execution** - Built-in support for standard library parallelism
- **Cache-friendly** - Optimized memory access patterns
- **Vectorization** - SIMD-friendly implementations

## 📊 Benchmarking Results

### Integrator Performance Comparison

| Method | Accuracy | Performance | Memory Usage | Best Use Case |
|--------|----------|-------------|--------------|---------------|
| Euler | Low | Fastest | Minimal | Simple systems, real-time |
| RK4 | Good | Fast | Low | General purpose |
| RK45 | Excellent | Adaptive | Medium | Smooth systems |
| DOP853 | Excellent | Adaptive | High | High precision required |
| BDF | Good | Stiff-optimized | Medium | Stiff systems |

## 🎯 Optimization Strategies

### 1. Choose the Right Integrator

```cpp
// For simple, non-stiff systems
diffeq::integrators::rk4<StateType> integrator;

// For stiff systems
diffeq::integrators::bdf<StateType> integrator;

// For high-precision requirements
diffeq::integrators::dop853<StateType> integrator;
integrator.set_tolerances(1e-12, 1e-10);

// For real-time applications
diffeq::integrators::euler<StateType> integrator;
```

### 2. Optimize State Types

```cpp
// Use stack-allocated arrays for small, fixed-size states
#include <array>
using StateType = std::array<double, 3>;

// Use Eigen for linear algebra operations
#include <Eigen/Dense>
using StateType = Eigen::Vector3d;

// Custom state types with optimized operations
struct OptimizedState {
    double x, y, z;
    
    // Inline operators for performance
    inline OptimizedState operator+(const OptimizedState& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }
    
    inline OptimizedState operator*(double scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }
};
```

## 🔄 Parallel Computing

### Standard Library Parallelism

```cpp
#include <execution>
#include <algorithm>

// Parallel integration of multiple systems
std::vector<std::vector<double>> systems(10000);
diffeq::integrators::rk4<std::vector<double>> integrator;

// Parallel for_each
std::for_each(std::execution::par_unseq,
              systems.begin(), systems.end(),
              [&](auto& state) {
                  double t = 0.0;
                  double dt = 0.01;
                  for (int i = 0; i < 1000; ++i) {
                      integrator.step(ode_system, state, t, dt);
                  }
              });
```

## 🔧 Compiler-Specific Optimizations

### GCC/Clang Optimizations

```bash
# Basic optimization flags
-O3 -march=native -DNDEBUG -flto

# Advanced optimization flags
-O3 -march=native -DNDEBUG -flto -ffast-math -funroll-loops
```

### MSVC Optimizations

```bash
# Basic optimization flags
/O2 /DNDEBUG /GL

# Advanced optimization flags
/O2 /Ox /DNDEBUG /GL /arch:AVX2 /fp:fast
```

## 📈 Performance Profiling

### Built-in Timing

```cpp
#include <chrono>

class Timer {
    std::chrono::high_resolution_clock::time_point start_;
    
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start_).count();
    }
};

// Usage
Timer timer;
integrator.integrate(system, state, t_start, t_end, dt);
std::cout << "Integration took " << timer.elapsed() << " seconds" << std::endl;
```

## 🔗 Performance Best Practices

### Do's
- ✅ Use appropriate integrator for your problem type
- ✅ Enable compiler optimizations (-O3, -march=native)
- ✅ Profile your code to identify bottlenecks
- ✅ Use parallel algorithms for independent systems
- ✅ Pre-allocate memory when possible
- ✅ Use stack allocation for small, fixed-size states
- ✅ Choose proper tolerances for adaptive methods

### Don'ts
- ❌ Don't use debug builds for performance measurements
- ❌ Don't allocate memory during integration loops
- ❌ Don't use overly tight tolerances unless necessary
- ❌ Don't share integrator instances between threads
- ❌ Don't ignore compiler warnings about vectorization
- ❌ Don't use std::vector for very small state sizes

## 🔗 See Also

- [API Documentation](../api/README.md) - Complete API reference
- [Examples](../examples/README.md) - Performance examples
- [Main Documentation](../index.md) - Library overview
- [Standard Parallelism Guide](../STANDARD_PARALLELISM.md) - Detailed parallelism information

## 📊 Hardware Recommendations

### CPU Requirements
- **Minimum**: Modern dual-core processor with C++20 support
- **Recommended**: Multi-core processor (6+ cores) for parallel workloads
- **Optimal**: High-end workstation or server with 16+ cores

### Memory Requirements
- **Minimum**: 4GB RAM for basic usage
- **Recommended**: 16GB+ RAM for large-scale simulations
- **Optimal**: 32GB+ RAM with fast memory (DDR4-3200 or higher)

### Compiler Support
- **GCC**: 11.0+ (full C++20 support)
- **Clang**: 13.0+ (full C++20 support)
- **MSVC**: 2022+ (Visual Studio 17.0)
- **Intel**: 2021.4+ (oneAPI DPC++/C++) 