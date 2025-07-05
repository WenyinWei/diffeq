# Integrating Standard Parallelism Libraries with diffeq

This document shows how to use standard parallelism libraries with the diffeq library, addressing the feedback to avoid custom parallel classes and use proven standard libraries instead.

## Quick Start: Simplified Interface

For most users, the simplest approach is to use the unified convenience interface:

```cpp
#include <diffeq.hpp>

// SIMPLEST USAGE: Just add parallel to existing code
auto system = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = y[1];           // dx/dt = v  
    dydt[1] = -y[0];          // dv/dt = -x
};

std::vector<std::vector<double>> states(100, {1.0, 0.0});  // 100 initial conditions

// THIS IS ALL YOU NEED! Automatic hardware selection and parallel execution:
diffeq::parallel::integrate_parallel(system, states, 0.01, 1000);
```

## Available Interfaces

### 1. **Simple Functions** (Recommended for most users)
- `diffeq::parallel::integrate_parallel()` - Automatic backend selection
- `diffeq::parallel::parameter_sweep_parallel()` - Parallel parameter sweeps
- `diffeq::parallel::create_async_dispatcher()` - One-by-one task processing

### 2. **Class-based Interface** (For advanced control)
- `diffeq::parallel::ODEParallel<State, Time>` - Unified parallel integrator
- Manual backend selection and configuration

### 3. **Direct Standard Libraries** (Maximum control)
- `diffeq::examples::ODEStdExecution` - C++17/20 std::execution
- `diffeq::examples::ODEOpenMP` - OpenMP parallel loops  
- `diffeq::examples::ODETBB` - Intel Threading Building Blocks
- `diffeq::examples::ODEThrust` - NVIDIA Thrust (GPU without kernels)
- `diffeq::examples::ODECuda` - Direct CUDA kernels
- `diffeq::examples::ODEOpenCL` - OpenCL cross-platform

Note: Class names use **suffix** naming (ODE + Backend) for better auto-completion grouping.

## Overview

Instead of creating custom parallel classes, we recommend using established standard libraries:

- **std::execution** - C++17/20 execution policies
- **OpenMP** - Cross-platform shared memory multiprocessing
- **Intel TBB** - Threading Building Blocks for advanced parallel algorithms
- **NVIDIA Thrust** - GPU acceleration without writing CUDA kernels

## Quick Examples

### 1. Basic Parallel Integration with std::execution

```cpp
#include <diffeq.hpp>
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
diffeq::examples::ODEStdExecution<std::vector<double>, double>::integrate_multiple_conditions(
    system, initial_conditions, 0.01, 100
);
```

### 2. Parameter Sweep (Beyond Initial Conditions)

```cpp
// Vary system parameters in parallel
std::vector<double> frequencies = {0.5, 1.0, 1.5, 2.0, 2.5};
std::vector<std::vector<double>> results;

diffeq::examples::ODEStdExecution<std::vector<double>, double>::parameter_sweep(
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

## Advanced GPU Computing: CUDA and OpenCL

### Direct CUDA Kernel Usage

For maximum GPU performance, you can write custom CUDA kernels:

```cpp
// CUDA kernel for parallel ODE integration
__global__ void integrate_ode_kernel(
    double* states,           // State vectors (flattened)
    double* parameters,       // System parameters
    double dt,
    int steps,
    int state_size,
    int num_systems
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_systems) return;
    
    // Each thread handles one ODE system
    double* my_state = &states[idx * state_size];
    double param = parameters[idx];
    
    for (int step = 0; step < steps; ++step) {
        // Simple harmonic oscillator: d²x/dt² = -ω²x
        double x = my_state[0];
        double v = my_state[1];
        double omega = param;
        
        // Euler integration (for simplicity)
        double new_v = v - omega * omega * x * dt;
        double new_x = x + v * dt;
        
        my_state[0] = new_x;
        my_state[1] = new_v;
    }
}

// Host function to launch CUDA kernel
void cuda_parallel_ode_integration(
    std::vector<std::vector<double>>& states,
    const std::vector<double>& parameters,
    double dt,
    int steps
) {
    // Flatten state data for GPU
    int num_systems = states.size();
    int state_size = states[0].size();
    
    std::vector<double> flat_states(num_systems * state_size);
    for (int i = 0; i < num_systems; ++i) {
        for (int j = 0; j < state_size; ++j) {
            flat_states[i * state_size + j] = states[i][j];
        }
    }
    
    // Allocate GPU memory
    double *d_states, *d_parameters;
    cudaMalloc(&d_states, num_systems * state_size * sizeof(double));
    cudaMalloc(&d_parameters, num_systems * sizeof(double));
    
    // Copy data to GPU
    cudaMemcpy(d_states, flat_states.data(), 
               num_systems * state_size * sizeof(double), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_parameters, parameters.data(), 
               num_systems * sizeof(double), 
               cudaMemcpyHostToDevice);
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (num_systems + block_size - 1) / block_size;
    integrate_ode_kernel<<<grid_size, block_size>>>(
        d_states, d_parameters, dt, steps, state_size, num_systems
    );
    
    // Copy results back
    cudaMemcpy(flat_states.data(), d_states, 
               num_systems * state_size * sizeof(double), 
               cudaMemcpyDeviceToHost);
    
    // Unflatten results
    for (int i = 0; i < num_systems; ++i) {
        for (int j = 0; j < state_size; ++j) {
            states[i][j] = flat_states[i * state_size + j];
        }
    }
    
    // Cleanup
    cudaFree(d_states);
    cudaFree(d_parameters);
}
```

### OpenCL for Cross-Platform GPU Computing

OpenCL provides broader hardware support (NVIDIA, AMD, Intel):

```cpp
#include <CL/cl.hpp>

// OpenCL kernel source code
const char* opencl_kernel_source = R"(
__kernel void integrate_ode_opencl(
    __global double* states,
    __global double* parameters,
    double dt,
    int steps,
    int state_size
) {
    int idx = get_global_id(0);
    
    // Each work-item handles one ODE system
    __global double* my_state = &states[idx * state_size];
    double param = parameters[idx];
    
    for (int step = 0; step < steps; ++step) {
        double x = my_state[0];
        double v = my_state[1];
        double omega = param;
        
        // Euler integration
        double new_v = v - omega * omega * x * dt;
        double new_x = x + v * dt;
        
        my_state[0] = new_x;
        my_state[1] = new_v;
    }
}
)";

void opencl_parallel_ode_integration(
    std::vector<std::vector<double>>& states,
    const std::vector<double>& parameters,
    double dt,
    int steps
) {
    // OpenCL setup
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    
    cl::Context context(CL_DEVICE_TYPE_GPU);
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::CommandQueue queue(context, devices[0]);
    
    // Create and build program
    cl::Program program(context, opencl_kernel_source);
    program.build(devices);
    cl::Kernel kernel(program, "integrate_ode_opencl");
    
    // Prepare data
    int num_systems = states.size();
    int state_size = states[0].size();
    std::vector<double> flat_states(num_systems * state_size);
    
    for (int i = 0; i < num_systems; ++i) {
        for (int j = 0; j < state_size; ++j) {
            flat_states[i * state_size + j] = states[i][j];
        }
    }
    
    // Create buffers
    cl::Buffer states_buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(double) * flat_states.size(), flat_states.data());
    cl::Buffer params_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(double) * parameters.size(), 
                            const_cast<double*>(parameters.data()));
    
    // Set kernel arguments
    kernel.setArg(0, states_buffer);
    kernel.setArg(1, params_buffer);
    kernel.setArg(2, dt);
    kernel.setArg(3, steps);
    kernel.setArg(4, state_size);
    
    // Execute kernel
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, 
                              cl::NDRange(num_systems), cl::NullRange);
    
    // Read results
    queue.enqueueReadBuffer(states_buffer, CL_TRUE, 0,
                           sizeof(double) * flat_states.size(), flat_states.data());
    
    // Unflatten results
    for (int i = 0; i < num_systems; ++i) {
        for (int j = 0; j < state_size; ++j) {
            states[i][j] = flat_states[i * state_size + j];
        }
    }
}
```

## Asynchronous CPU Processing

### One-by-One Thread Startup with std::async

For scenarios where trigger signals arrive one-by-one:

```cpp
#include <future>
#include <queue>
#include <thread>
#include <condition_variable>
#include <mutex>

class ODETaskDispatcher {
private:
    std::queue<std::function<void()>> task_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> stop_flag{false};
    
public:
    // Submit ODE integration task asynchronously
    template<typename System>
    std::future<std::vector<double>> submit_ode_task(
        System&& system,
        std::vector<double> initial_state,
        double dt,
        int steps
    ) {
        auto task = std::make_shared<std::packaged_task<std::vector<double>()>>(
            [system = std::forward<System>(system), initial_state, dt, steps]() mutable {
                auto integrator = diffeq::integrators::ode::RK4Integrator<std::vector<double>, double>(system);
                std::vector<double> state = initial_state;
                
                for (int i = 0; i < steps; ++i) {
                    integrator.step(state, dt);
                }
                return state;
            }
        );
        
        auto future = task->get_future();
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_queue.emplace([task]() { (*task)(); });
        }
        cv.notify_one();
        
        return future;
    }
    
    // Process tasks one-by-one as they arrive
    void process_tasks() {
        while (!stop_flag) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            cv.wait(lock, [this] { return !task_queue.empty() || stop_flag; });
            
            if (stop_flag) break;
            
            auto task = std::move(task_queue.front());
            task_queue.pop();
            lock.unlock();
            
            // Execute task in new thread (started one-by-one)
            std::thread(task).detach();
        }
    }
    
    void stop() {
        stop_flag = true;
        cv.notify_all();
    }
};

// Usage example
void demonstrate_async_processing() {
    ODETaskDispatcher processor;
    
    // Start the processor in background
    std::thread processor_thread(&ODETaskDispatcher::process_tasks, &processor);
    
    // Simple harmonic oscillator
    auto system = [](const std::vector<double>& y, std::vector<double>& dydt, double t) {
        dydt[0] = y[1];
        dydt[1] = -y[0];  // ω = 1
    };
    
    std::vector<std::future<std::vector<double>>> futures;
    
    // Submit tasks one-by-one as signals arrive
    for (int i = 0; i < 10; ++i) {
        // Simulate trigger signal arrival
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        std::vector<double> initial_state = {static_cast<double>(i), 0.0};
        auto future = processor.submit_ode_task(system, initial_state, 0.01, 100);
        futures.push_back(std::move(future));
        
        std::cout << "Submitted task " << i << " (triggered by signal)\n";
    }
    
    // Collect results
    for (size_t i = 0; i < futures.size(); ++i) {
        auto result = futures[i].get();
        std::cout << "Task " << i << " completed: x=" << result[0] << std::endl;
    }
    
    processor.stop();
    processor_thread.join();
}
```

### Process-Based Asynchronous Computing

For distributed computing across multiple processes:

```cpp
#include <mpi.h>

class MPIODETaskDispatcher {
public:
    static void master_process() {
        int world_size, world_rank;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        
        std::vector<int> worker_status(world_size - 1, 0); // 0 = idle, 1 = busy
        
        // Distribute tasks to workers as signals arrive
        for (int task_id = 0; task_id < 20; ++task_id) {
            // Wait for trigger signal (simulated)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            
            // Find available worker
            int worker = -1;
            for (int i = 1; i < world_size; ++i) {
                if (worker_status[i-1] == 0) {
                    worker = i;
                    worker_status[i-1] = 1;
                    break;
                }
            }
            
            if (worker != -1) {
                // Send task to worker
                double initial_condition = task_id * 0.1;
                MPI_Send(&initial_condition, 1, MPI_DOUBLE, worker, 0, MPI_COMM_WORLD);
                std::cout << "Sent task " << task_id << " to worker " << worker << std::endl;
            } else {
                // No workers available, queue task or wait
                task_id--; // Retry this task
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            // Check for completed tasks
            MPI_Status status;
            int flag;
            MPI_Iprobe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &flag, &status);
            if (flag) {
                double result;
                MPI_Recv(&result, 1, MPI_DOUBLE, status.MPI_SOURCE, 1, MPI_COMM_WORLD, &status);
                worker_status[status.MPI_SOURCE - 1] = 0; // Mark worker as idle
                std::cout << "Received result " << result << " from worker " << status.MPI_SOURCE << std::endl;
            }
        }
        
        // Signal workers to stop
        for (int i = 1; i < world_size; ++i) {
            double stop_signal = -1.0;
            MPI_Send(&stop_signal, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    }
    
    static void worker_process() {
        while (true) {
            double initial_condition;
            MPI_Status status;
            MPI_Recv(&initial_condition, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            
            if (initial_condition < 0) break; // Stop signal
            
            // Perform ODE integration
            std::vector<double> state = {initial_condition, 0.0};
            auto system = [](const std::vector<double>& y, std::vector<double>& dydt, double t) {
                dydt[0] = y[1];
                dydt[1] = -y[0];
            };
            
            auto integrator = diffeq::integrators::ode::RK4Integrator<std::vector<double>, double>(system);
            for (int i = 0; i < 100; ++i) {
                integrator.step(state, 0.01);
            }
            
            // Send result back
            MPI_Send(&state[0], 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }
    }
};
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

# For direct CUDA kernel usage
enable_language(CUDA)
set_property(TARGET your_target PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# For OpenCL
find_package(OpenCL REQUIRED)
target_link_libraries(your_target OpenCL::OpenCL)

# For MPI (distributed computing)
find_package(MPI REQUIRED)
target_link_libraries(your_target MPI::MPI_CXX)
```

## Summary: Choosing the Right Approach

| Use Case | Recommended Approach | Why |
|----------|---------------------|-----|
| Simple parallel loops | `std::execution::par` | Built-in, no dependencies |
| CPU-intensive work | OpenMP | Mature, cross-platform |
| Complex task scheduling | Intel TBB | Advanced algorithms |
| Maximum GPU performance | Direct CUDA kernels | Full control, optimal performance |
| Cross-platform GPU | OpenCL | Broad hardware support |
| GPU without kernel writing | NVIDIA Thrust | Ease of use |
| One-by-one task arrival | `std::async` with queues | Event-driven processing |
| Distributed computing | MPI + async patterns | Scale across multiple machines |

This approach gives you maximum flexibility while leveraging proven, optimized libraries instead of reinventing the wheel.