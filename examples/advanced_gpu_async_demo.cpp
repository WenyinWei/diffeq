#include <iostream>
#include <vector>
#include <future>
#include <queue>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <chrono>
#include <functional>

/**
 * @brief Advanced GPU and Asynchronous CPU Processing Demo
 * 
 * This demonstrates:
 * 1. Direct CUDA kernel usage for ODE integration
 * 2. OpenCL cross-platform GPU computing
 * 3. Asynchronous CPU processing with one-by-one thread startup
 * 4. Process-based distributed computing patterns
 */

// ================================================================
// CUDA Direct Kernel Usage
// ================================================================
#ifdef __CUDACC__
#include <cuda_runtime.h>

// CUDA kernel for parallel ODE integration
__global__ void integrate_harmonic_oscillator_cuda(
    double* states,           // Flattened state vectors [x0, v0, x1, v1, ...]
    double* frequencies,      // Frequency parameters for each oscillator
    double dt,
    int steps,
    int num_systems
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_systems) return;
    
    // Each thread handles one harmonic oscillator
    double x = states[2 * idx];
    double v = states[2 * idx + 1];
    double omega = frequencies[idx];
    
    // Integrate using Verlet method for better stability
    for (int step = 0; step < steps; ++step) {
        double a = -omega * omega * x;  // acceleration = -ω²x
        double new_x = x + v * dt + 0.5 * a * dt * dt;
        double new_a = -omega * omega * new_x;
        double new_v = v + 0.5 * (a + new_a) * dt;
        
        x = new_x;
        v = new_v;
    }
    
    states[2 * idx] = x;
    states[2 * idx + 1] = v;
}

class CUDAODEIntegrator {
public:
    static bool cuda_available() {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        return (error == cudaSuccess) && (device_count > 0);
    }
    
    static void integrate_harmonic_oscillators(
        std::vector<std::vector<double>>& states,
        const std::vector<double>& frequencies,
        double dt,
        int steps
    ) {
        int num_systems = states.size();
        
        // Flatten state data for GPU
        std::vector<double> flat_states(num_systems * 2);
        for (int i = 0; i < num_systems; ++i) {
            flat_states[2 * i] = states[i][0];     // position
            flat_states[2 * i + 1] = states[i][1]; // velocity
        }
        
        // Allocate GPU memory
        double *d_states, *d_frequencies;
        cudaMalloc(&d_states, num_systems * 2 * sizeof(double));
        cudaMalloc(&d_frequencies, num_systems * sizeof(double));
        
        // Copy data to GPU
        cudaMemcpy(d_states, flat_states.data(), 
                   num_systems * 2 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frequencies, frequencies.data(), 
                   num_systems * sizeof(double), cudaMemcpyHostToDevice);
        
        // Launch kernel
        int block_size = 256;
        int grid_size = (num_systems + block_size - 1) / block_size;
        
        integrate_harmonic_oscillator_cuda<<<grid_size, block_size>>>(
            d_states, d_frequencies, dt, steps, num_systems
        );
        
        cudaDeviceSynchronize();
        
        // Copy results back
        cudaMemcpy(flat_states.data(), d_states, 
                   num_systems * 2 * sizeof(double), cudaMemcpyDeviceToHost);
        
        // Unflatten results
        for (int i = 0; i < num_systems; ++i) {
            states[i][0] = flat_states[2 * i];
            states[i][1] = flat_states[2 * i + 1];
        }
        
        // Cleanup
        cudaFree(d_states);
        cudaFree(d_frequencies);
    }
};
#endif

// ================================================================
// OpenCL Cross-Platform GPU Computing
// ================================================================
#ifdef OPENCL_AVAILABLE
#include <CL/cl.hpp>

const char* opencl_harmonic_oscillator_kernel = R"(
__kernel void integrate_harmonic_oscillator_opencl(
    __global double* states,
    __global double* frequencies,
    double dt,
    int steps
) {
    int idx = get_global_id(0);
    
    // Each work-item handles one harmonic oscillator
    double x = states[2 * idx];
    double v = states[2 * idx + 1];
    double omega = frequencies[idx];
    
    // Verlet integration for stability
    for (int step = 0; step < steps; ++step) {
        double a = -omega * omega * x;
        double new_x = x + v * dt + 0.5 * a * dt * dt;
        double new_a = -omega * omega * new_x;
        double new_v = v + 0.5 * (a + new_a) * dt;
        
        x = new_x;
        v = new_v;
    }
    
    states[2 * idx] = x;
    states[2 * idx + 1] = v;
}
)";

class ODEOpenCLIntegrator {
public:
    static bool opencl_available() {
        try {
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            return !platforms.empty();
        } catch (...) {
            return false;
        }
    }
    
    static void integrate_harmonic_oscillators(
        std::vector<std::vector<double>>& states,
        const std::vector<double>& frequencies,
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
        cl::Program program(context, opencl_harmonic_oscillator_kernel);
        program.build(devices);
        cl::Kernel kernel(program, "integrate_harmonic_oscillator_opencl");
        
        // Prepare data
        int num_systems = states.size();
        std::vector<double> flat_states(num_systems * 2);
        for (int i = 0; i < num_systems; ++i) {
            flat_states[2 * i] = states[i][0];
            flat_states[2 * i + 1] = states[i][1];
        }
        
        // Create buffers
        cl::Buffer states_buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(double) * flat_states.size(), flat_states.data());
        cl::Buffer freq_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(double) * frequencies.size(), 
                              const_cast<double*>(frequencies.data()));
        
        // Set kernel arguments
        kernel.setArg(0, states_buffer);
        kernel.setArg(1, freq_buffer);
        kernel.setArg(2, dt);
        kernel.setArg(3, steps);
        
        // Execute kernel
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, 
                                  cl::NDRange(num_systems), cl::NullRange);
        
        // Read results
        queue.enqueueReadBuffer(states_buffer, CL_TRUE, 0,
                               sizeof(double) * flat_states.size(), flat_states.data());
        
        // Unflatten results
        for (int i = 0; i < num_systems; ++i) {
            states[i][0] = flat_states[2 * i];
            states[i][1] = flat_states[2 * i + 1];
        }
    }
};
#endif

// ================================================================
// Asynchronous CPU Processing - One-by-One Thread Startup
// ================================================================

class ODETaskDispatcher {
private:
    std::queue<std::function<void()>> task_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> stop_flag{false};
    std::vector<std::thread> worker_threads;
    
public:
    ODETaskDispatcher() = default;
    
    ~ODETaskDispatcher() {
        stop();
    }
    
    // Submit ODE integration task that will start when triggered
    std::future<std::vector<double>> submit_harmonic_oscillator_task(
        double omega,
        std::vector<double> initial_state,
        double dt,
        int steps
    ) {
        auto task = std::make_shared<std::packaged_task<std::vector<double>()>>(
            [omega, initial_state, dt, steps]() {
                std::vector<double> state = initial_state;
                
                // Simple harmonic oscillator integration
                for (int i = 0; i < steps; ++i) {
                    double x = state[0];
                    double v = state[1];
                    
                    // Verlet integration
                    double a = -omega * omega * x;
                    double new_x = x + v * dt + 0.5 * a * dt * dt;
                    double new_a = -omega * omega * new_x;
                    double new_v = v + 0.5 * (a + new_a) * dt;
                    
                    state[0] = new_x;
                    state[1] = new_v;
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
    
    // Start processing - each task gets its own thread when triggered
    void start_async_processing() {
        // Single dispatcher thread that spawns worker threads one-by-one
        worker_threads.emplace_back([this]() {
            while (!stop_flag) {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv.wait(lock, [this] { return !task_queue.empty() || stop_flag; });
                
                if (stop_flag) break;
                
                auto task = std::move(task_queue.front());
                task_queue.pop();
                lock.unlock();
                
                // Start new thread for this specific task (one-by-one startup)
                std::thread worker_thread(task);
                worker_thread.detach(); // Let it run independently
                
                std::cout << "Started new thread for incoming task\n";
            }
        });
    }
    
    void stop() {
        stop_flag = true;
        cv.notify_all();
        
        for (auto& thread : worker_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        worker_threads.clear();
    }
};

// ================================================================
// Signal-Driven Processing Example
// ================================================================

class SignalDrivenODEProcessor {
private:
    ODETaskDispatcher processor;
    std::atomic<int> signal_count{0};
    
public:
    void start() {
        processor.start_async_processing();
    }
    
    void stop() {
        processor.stop();
    }
    
    // Simulate external signal triggering ODE computation
    void handle_external_signal(double omega, std::vector<double> initial_state) {
        int signal_id = signal_count++;
        
        std::cout << "Signal " << signal_id << " received - triggering ODE computation\n";
        
        auto future = processor.submit_harmonic_oscillator_task(
            omega, initial_state, 0.01, 100
        );
        
        // Could store future for later retrieval or process immediately
        std::thread([future = std::move(future), signal_id]() mutable {
            auto result = future.get();
            std::cout << "Signal " << signal_id << " completed: x=" << result[0] 
                      << ", v=" << result[1] << std::endl;
        }).detach();
    }
};

// ================================================================
// Demo Functions
// ================================================================

void demonstrate_cuda_direct() {
    std::cout << "\n=== CUDA Direct Kernel Usage ===\n";
    
#ifdef __CUDACC__
    if (CUDAODEIntegrator::cuda_available()) {
        std::cout << "CUDA GPU detected!\n";
        
        // Setup harmonic oscillators with different frequencies
        const int num_oscillators = 1000;
        std::vector<std::vector<double>> states(num_oscillators, {1.0, 0.0}); // x=1, v=0
        std::vector<double> frequencies;
        
        for (int i = 0; i < num_oscillators; ++i) {
            frequencies.push_back(0.1 + i * 0.01); // ω = 0.1, 0.11, 0.12, ...
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        CUDAODEIntegrator::integrate_harmonic_oscillators(states, frequencies, 0.01, 1000);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "CUDA integration of " << num_oscillators << " systems: " 
                  << duration.count() << " μs\n";
        std::cout << "First oscillator result: x=" << states[0][0] << ", v=" << states[0][1] << "\n";
    } else {
        std::cout << "CUDA not available\n";
    }
#else
    std::cout << "Not compiled with CUDA support\n";
#endif
}

void demonstrate_opencl() {
    std::cout << "\n=== OpenCL Cross-Platform GPU Computing ===\n";
    
#ifdef OPENCL_AVAILABLE
    if (ODEOpenCLIntegrator::opencl_available()) {
        std::cout << "OpenCL devices detected!\n";
        
        const int num_oscillators = 500;
        std::vector<std::vector<double>> states(num_oscillators, {1.0, 0.0});
        std::vector<double> frequencies;
        
        for (int i = 0; i < num_oscillators; ++i) {
            frequencies.push_back(0.5 + i * 0.01);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        ODEOpenCLIntegrator::integrate_harmonic_oscillators(states, frequencies, 0.01, 1000);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "OpenCL integration of " << num_oscillators << " systems: " 
                  << duration.count() << " μs\n";
        std::cout << "First oscillator result: x=" << states[0][0] << ", v=" << states[0][1] << "\n";
    } else {
        std::cout << "OpenCL not available\n";
    }
#else
    std::cout << "Not compiled with OpenCL support\n";
#endif
}

void demonstrate_async_processing() {
    std::cout << "\n=== Asynchronous One-by-One Thread Startup ===\n";
    
    SignalDrivenODEProcessor processor;
    processor.start();
    
    // Simulate external signals arriving one-by-one
    std::vector<std::future<void>> signal_futures;
    
    for (int i = 0; i < 10; ++i) {
        // Each signal arrives at different times (simulating real-world scenario)
        signal_futures.push_back(std::async(std::launch::async, [&processor, i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(i * 50));
            
            double omega = 1.0 + i * 0.1;
            std::vector<double> initial_state = {static_cast<double>(i), 0.0};
            
            processor.handle_external_signal(omega, initial_state);
        }));
    }
    
    // Wait for all signals to be processed
    for (auto& future : signal_futures) {
        future.wait();
    }
    
    // Allow time for processing to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    processor.stop();
    std::cout << "Asynchronous processing demonstration completed\n";
}

void demonstrate_performance_comparison() {
    std::cout << "\n=== Performance Comparison ===\n";
    
    const int num_systems = 1000;
    const int steps = 500;
    const double dt = 0.01;
    
    // Setup identical test data
    std::vector<std::vector<double>> states_cpu(num_systems, {1.0, 0.0});
    std::vector<std::vector<double>> states_gpu = states_cpu;
    std::vector<double> frequencies;
    
    for (int i = 0; i < num_systems; ++i) {
        frequencies.push_back(0.1 + i * 0.001);
    }
    
    // CPU benchmark (sequential)
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_systems; ++i) {
        double omega = frequencies[i];
        for (int step = 0; step < steps; ++step) {
            double x = states_cpu[i][0];
            double v = states_cpu[i][1];
            double a = -omega * omega * x;
            double new_x = x + v * dt + 0.5 * a * dt * dt;
            double new_a = -omega * omega * new_x;
            double new_v = v + 0.5 * (a + new_a) * dt;
            states_cpu[i][0] = new_x;
            states_cpu[i][1] = new_v;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "CPU Sequential: " << cpu_duration.count() << " μs\n";
    
#ifdef __CUDACC__
    if (CUDAODEIntegrator::cuda_available()) {
        start = std::chrono::high_resolution_clock::now();
        CUDAODEIntegrator::integrate_harmonic_oscillators(states_gpu, frequencies, dt, steps);
        end = std::chrono::high_resolution_clock::now();
        auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "GPU CUDA: " << gpu_duration.count() << " μs\n";
        std::cout << "Speedup: " << static_cast<double>(cpu_duration.count()) / gpu_duration.count() << "x\n";
        
        // Verify results match
        double max_error = 0.0;
        for (int i = 0; i < num_systems; ++i) {
            double error = std::abs(states_cpu[i][0] - states_gpu[i][0]);
            max_error = std::max(max_error, error);
        }
        std::cout << "Maximum numerical difference: " << max_error << "\n";
    }
#endif
}

int main() {
    std::cout << "Advanced GPU and Asynchronous CPU Processing Demo\n";
    std::cout << "================================================\n";
    
    std::cout << "Available features:\n";
#ifdef __CUDACC__
    std::cout << "✓ CUDA direct kernel support\n";
#else
    std::cout << "✗ CUDA direct kernel support (compile with nvcc)\n";
#endif

#ifdef OPENCL_AVAILABLE
    std::cout << "✓ OpenCL cross-platform support\n";
#else
    std::cout << "✗ OpenCL cross-platform support\n";
#endif

    std::cout << "✓ Asynchronous CPU processing\n";
    std::cout << "✓ Signal-driven computation\n";
    
    try {
        demonstrate_cuda_direct();
        demonstrate_opencl();
        demonstrate_async_processing();
        demonstrate_performance_comparison();
        
        std::cout << "\n=== Summary ===\n";
        std::cout << "✅ Direct CUDA kernels: Maximum GPU performance with full control\n";
        std::cout << "✅ OpenCL: Cross-platform GPU/CPU acceleration\n";
        std::cout << "✅ Async processing: One-by-one thread startup for signal-driven scenarios\n";
        std::cout << "✅ Performance comparison: Quantified speedups and accuracy\n";
        std::cout << "\nUsers can now:\n";
        std::cout << "• Write custom CUDA kernels for maximum performance\n";
        std::cout << "• Use OpenCL for broader hardware support\n";
        std::cout << "• Handle asynchronous signal-driven ODE computations\n";
        std::cout << "• Choose the optimal approach for their specific use case\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}