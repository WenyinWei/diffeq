# Integrating Standard Parallelism Libraries with diffeq

This document shows how to use standard parallelism libraries with the diffeq library, addressing the feedback to avoid custom parallel classes and use proven standard libraries instead.

## Overview

Instead of creating custom parallel classes, we recommend using established standard libraries:

- **std::execution** - C++17/20 execution policies
- **OpenMP** - Cross-platform shared memory multiprocessing
- **Intel TBB** - Threading Building Blocks for advanced parallel algorithms
- **NVIDIA Thrust** - GPU acceleration without writing CUDA kernels

## Quick Examples

### 1. Basic Parallel Integration with std::execution

```cpp
#include <examples/standard_parallelism.hpp>
#include <execution>
#include <algorithm>

// Simple harmonic oscillator
struct HarmonicOscillator {
    void operator()(const std::vector<double>& y, std::vector<double>& dydt, double t) const {
        dydt[0] = y[1];           // dx/dt = v
        dydt[1] = -y[0];          // dv/dt = -x (ω=1)
    }
};

// Multiple initial conditions in parallel
std::vector<std::vector<double>> initial_conditions(1000);
// ... fill with different initial conditions ...

HarmonicOscillator system;
diffeq::examples::StandardParallelODE<std::vector<double>, double>::integrate_multiple_conditions(
    system, initial_conditions, 0.01, 100
);
```

### 2. Parameter Sweep (Beyond Initial Conditions)

```cpp
// Vary system parameters in parallel
std::vector<double> frequencies = {0.5, 1.0, 1.5, 2.0, 2.5};
std::vector<std::vector<double>> results;

diffeq::examples::StandardParallelODE<std::vector<double>, double>::parameter_sweep(
    [](const std::vector<double>& y, std::vector<double>& dydt, double t, double omega) {
        dydt[0] = y[1];
        dydt[1] = -omega*omega*y[0];  // Parameterized frequency
    },
    {1.0, 0.0}, frequencies, results, 0.01, 100
);
```

### 3. OpenMP for CPU Parallelism

```cpp
#include <omp.h>

std::vector<std::vector<double>> states(1000);
// ... initialize states ...

#pragma omp parallel for
for (size_t i = 0; i < states.size(); ++i) {
    auto integrator = diffeq::integrators::ode::RK4Integrator<std::vector<double>, double>(system);
    for (int step = 0; step < 100; ++step) {
        integrator.step(states[i], 0.01);
    }
}
```

### 4. GPU Acceleration with Thrust (NO Custom Kernels!)

```cpp
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

// Copy to GPU
thrust::device_vector<std::vector<double>> gpu_states = host_states;

// GPU parallel execution without writing kernels!
thrust::for_each(thrust::device, gpu_states.begin(), gpu_states.end(),
    [=] __device__ (std::vector<double>& state) {
        // Your ODE integration code here
        // (integrator needs to be GPU-compatible)
    });

// Copy back to host
thrust::copy(gpu_states.begin(), gpu_states.end(), host_states.begin());
```

### 5. Intel TBB for Advanced Parallelism

```cpp
#include <tbb/parallel_for_each.h>

tbb::parallel_for_each(states.begin(), states.end(),
    [&](std::vector<double>& state) {
        auto integrator = diffeq::integrators::ode::RK4Integrator<std::vector<double>, double>(system);
        for (int step = 0; step < 100; ++step) {
            integrator.step(state, 0.01);
        }
    });
```

## Benefits of This Approach

### ✅ Advantages
- **Proven Libraries**: Use optimized, well-tested standard libraries
- **No Learning Curve**: No custom classes to learn
- **Flexibility**: Vary parameters, integrators, callbacks, devices
- **Hardware Specific**: Choose the right tool for each use case
- **GPU Support**: Thrust provides GPU acceleration without writing kernels
- **Standard Detection**: Use standard library functions for hardware detection

### ❌ What We Avoid
- Custom "Facade" classes
- Reinventing parallel algorithms
- Custom hardware detection code
- Restricting flexibility to only initial conditions

## Choosing the Right Library

| Use Case | Recommended Library | Why |
|----------|-------------------|-----|
| Simple parallel loops | `std::execution::par` | Built into C++17, no dependencies |
| CPU-intensive computation | OpenMP | Mature, cross-platform, great CPU scaling |
| Complex task dependencies | Intel TBB | Advanced algorithms, work-stealing |
| GPU acceleration | NVIDIA Thrust | GPU without custom kernels |
| Mixed workloads | Combination | Use the right tool for each part |

## Real-World Examples

### Monte Carlo Simulations
```cpp
// Parameter sweep with different random seeds
std::vector<int> seeds(1000);
std::iota(seeds.begin(), seeds.end(), 1);

std::for_each(std::execution::par, seeds.begin(), seeds.end(),
    [&](int seed) {
        std::mt19937 rng(seed);
        // Run simulation with this random number generator
        // Each thread gets its own RNG state
    });
```

### Robotics Control Systems
```cpp
// Real-time control with different controller parameters
#pragma omp parallel for schedule(static)
for (size_t i = 0; i < control_parameters.size(); ++i) {
    auto controller = create_controller(control_parameters[i]);
    auto integrator = diffeq::integrators::ode::RK4Integrator<State, double>(controller);
    
    // Simulate control system
    for (int step = 0; step < simulation_steps; ++step) {
        integrator.step(robot_state[i], dt);
    }
}
```

### Multi-Physics Simulations
```cpp
// Different physics models running simultaneously
#pragma omp parallel sections
{
    #pragma omp section
    {
        // Fluid dynamics
        integrate_fluid_system();
    }
    
    #pragma omp section
    {
        // Structural mechanics
        integrate_structural_system();
    }
    
    #pragma omp section
    {
        // Heat transfer
        integrate_thermal_system();
    }
}
```

## Hardware Detection (Standard Way)

Instead of custom hardware detection, use standard library functions:

```cpp
// OpenMP thread count
int num_threads = omp_get_max_threads();

// CUDA device count
int device_count = 0;
cudaGetDeviceCount(&device_count);
bool gpu_available = (device_count > 0);

// TBB automatic initialization
tbb::task_scheduler_init init; // Uses all available cores
```

## Integration with Existing diffeq Code

The beauty of this approach is that **your existing diffeq code doesn't change**:

```cpp
// This works exactly as before
auto integrator = diffeq::integrators::ode::RK4Integrator<std::vector<double>, double>(system);
integrator.step(state, dt);

// Just wrap it in standard parallel constructs when you need parallelism
std::for_each(std::execution::par, states.begin(), states.end(),
    [&](auto& state) {
        auto integrator = diffeq::integrators::ode::RK4Integrator<std::vector<double>, double>(system);
        integrator.step(state, dt);
    });
```

## Building and Dependencies

### CMake Integration
```cmake
# For std::execution
target_compile_features(your_target PRIVATE cxx_std_17)

# For OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(your_target OpenMP::OpenMP_CXX)

# For Intel TBB
find_package(TBB REQUIRED)
target_link_libraries(your_target TBB::tbb)

# For NVIDIA Thrust (comes with CUDA)
find_package(CUDA REQUIRED)
target_link_libraries(your_target ${CUDA_LIBRARIES})
```

This approach gives you maximum flexibility while leveraging proven, optimized libraries instead of reinventing the wheel.