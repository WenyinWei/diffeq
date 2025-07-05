# Simplified Parallelism Usage in diffeq

The diffeq library now provides a much simpler parallelism interface alongside the advanced facade system.

## Quick Start (Simple Interface)

For most users, parallel execution is now as simple as:

```cpp
#include <diffeq.hpp>

// Parallel execution of ODE integration for multiple initial conditions
std::vector<std::vector<double>> initial_conditions = /* your data */;

diffeq::execution::parallel_for_each(initial_conditions, [](std::vector<double>& state) {
    auto integrator = diffeq::RK4Integrator<std::vector<double>>(your_system);
    integrator.step(state, dt);
});
```

## Simple API Reference

### Free Functions (Global)
```cpp
// Execute function on each element in parallel
diffeq::execution::parallel_for_each(container, lambda);

// Async execution with future
auto future = diffeq::execution::parallel_async(lambda);

// Configure global parallel settings
diffeq::execution::set_parallel_workers(8);
diffeq::execution::enable_gpu_acceleration();
diffeq::execution::enable_cpu_only();
```

### Parallel Class (Custom Instances)
```cpp
// Create custom parallel executor
auto parallel = diffeq::execution::Parallel(4); // 4 workers

// Execute on each element
parallel.for_each(container, lambda);

// Async execution
auto future = parallel.async(lambda);

// Configuration
parallel.set_workers(8);
parallel.use_gpu();
parallel.use_cpu();

// Information
size_t workers = parallel.worker_count();
bool gpu_available = parallel.gpu_available();
```

## Advanced Usage

For complex scenarios requiring fine-grained control, real-time priorities, hardware-specific optimizations, etc., the full ParallelismFacade and builder pattern are still available:

```cpp
auto facade = diffeq::execution::parallel_execution()
    .target_gpu()
    .use_thread_pool()
    .workers(1024)
    .realtime_priority()
    .enable_load_balancing()
    .build();
```

## Migration Guide

**Before (complex):**
```cpp
diffeq::execution::ParallelismFacade facade;
facade.parallel_for_each(container, lambda);
```

**Now (simple):**
```cpp
diffeq::execution::parallel_for_each(container, lambda);
```

The simplified interface makes parallelism accessible to all users while maintaining full backward compatibility.