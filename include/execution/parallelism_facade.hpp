#pragma once

#include <core/concepts.hpp>
#include <memory>
#include <future>
#include <vector>
#include <functional>
#include <type_traits>
#include <thread>
#include <iostream>

namespace diffeq::execution {

/**
 * @brief Hardware target types for parallel execution
 */
enum class HardwareTarget {
    CPU_Sequential,    // Single-threaded CPU execution
    CPU_OpenMP,        // OpenMP parallel CPU execution
    CPU_MPI,           // MPI distributed CPU execution
    CPU_ThreadPool,    // Thread pool CPU execution
    GPU_CUDA,          // NVIDIA CUDA GPU execution
    GPU_OpenCL,        // OpenCL GPU execution
    FPGA_HLS,          // High-Level Synthesis FPGA execution
    Auto               // Automatic hardware selection
};

/**
 * @brief Parallel computing paradigms
 */
enum class ParallelParadigm {
    ProcessPool,       // Multiple processes
    ThreadPool,        // Multiple threads
    Fibers,            // Lightweight fibers/coroutines
    SIMD,              // Single Instruction Multiple Data
    TaskGraph,         // Task dependency graph execution
    Pipeline           // Pipeline parallel execution
};

/**
 * @brief Configuration for parallel execution
 */
struct ParallelConfig {
    HardwareTarget target = HardwareTarget::Auto;
    ParallelParadigm paradigm = ParallelParadigm::ThreadPool;
    size_t max_workers = std::thread::hardware_concurrency();
    size_t batch_size = 1000;
    Priority priority = Priority::Normal;
    bool enable_load_balancing = true;
    bool enable_numa_awareness = false;
    
    // GPU-specific settings
    struct {
        int device_id = 0;
        size_t block_size = 256;
        size_t grid_size = 0;  // 0 = auto-calculate
    } gpu;
    
    // MPI-specific settings
    struct {
        int rank = 0;
        int size = 1;
        bool use_async_comm = true;
    } mpi;
};

/**
 * @brief Abstract base for hardware-specific execution strategies
 */
class ExecutionStrategy {
public:
    virtual ~ExecutionStrategy() = default;
    
    virtual std::future<void> execute_void(std::function<void()> func) = 0;
    
    template<typename F, typename... Args>
    auto execute(F&& func, Args&&... args) 
        -> std::future<std::invoke_result_t<F, Args...>> {
        
        using return_type = std::invoke_result_t<F, Args...>;
        
        if constexpr (std::is_void_v<return_type>) {
            return execute_void([=]() { func(args...); });
        } else {
            auto promise = std::make_shared<std::promise<return_type>>();
            auto future = promise->get_future();
            
            execute_void([=]() {
                try {
                    auto result = func(args...);
                    promise->set_value(result);
                } catch (...) {
                    promise->set_exception(std::current_exception());
                }
            });
            
            return future;
        }
    }
    
    virtual void bulk_execute_void(size_t count, std::function<void(size_t)> func) = 0;
    
    template<typename Iterator, typename F>
    void bulk_execute(Iterator first, Iterator last, F&& func) {
        size_t count = std::distance(first, last);
        bulk_execute_void(count, [=, &func](size_t index) {
            auto it = first;
            std::advance(it, index);
            func(*it);
        });
    }
    
    virtual bool is_available() const = 0;
    virtual HardwareTarget get_target() const = 0;
    virtual size_t get_max_concurrency() const = 0;
};

/**
 * @brief CPU-based execution strategies
 */
class CPUStrategy : public ExecutionStrategy {
private:
    std::shared_ptr<AdvancedExecutor> executor_;
    ParallelConfig config_;
    
public:
    explicit CPUStrategy(const ParallelConfig& config);
    
    std::future<void> execute_void(std::function<void()> func) override {
        auto task = std::make_shared<std::packaged_task<void()>>(std::move(func));
        auto future = task->get_future();
        
        // Use the basic submit method from AdvancedExecutor
        // For now, we'll simulate the execution
        std::thread([task]() { (*task)(); }).detach();
        
        return future;
    }
    
    void bulk_execute_void(size_t count, std::function<void(size_t)> func) override {
        const size_t num_threads = std::min(count, config_.max_workers);
        const size_t chunk_size = count / num_threads;
        
        std::vector<std::thread> threads;
        
        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = (t == num_threads - 1) ? count : (t + 1) * chunk_size;
            
            threads.emplace_back([=, &func]() {
                for (size_t i = start; i < end; ++i) {
                    func(i);
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    bool is_available() const override { return true; }
    HardwareTarget get_target() const override { return HardwareTarget::CPU_ThreadPool; }
    size_t get_max_concurrency() const override { return config_.max_workers; }
};

/**
 * @brief GPU-based execution strategy (CUDA/OpenCL)
 */
class GPUStrategy : public ExecutionStrategy {
private:
    ParallelConfig config_;
    bool cuda_available_;
    bool opencl_available_;
    
public:
    explicit GPUStrategy(const ParallelConfig& config);
    
    std::future<void> execute_void(std::function<void()> func) override {
        // For now, fallback to CPU execution with a warning
        std::cerr << "Warning: GPU execution not fully implemented, falling back to CPU\n";
        
        auto promise = std::make_shared<std::promise<void>>();
        auto future = promise->get_future();
        
        std::thread([=, func = std::move(func)]() mutable {
            try {
                func();
                promise->set_value();
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        }).detach();
        
        return future;
    }
    
    void bulk_execute_void(size_t count, std::function<void(size_t)> func) override {
        // For now, fallback to CPU execution
        std::cerr << "Warning: GPU bulk execution not fully implemented, falling back to CPU\n";
        
        std::vector<std::thread> threads;
        const size_t num_threads = std::min(count, config_.max_workers);
        const size_t chunk_size = count / num_threads;
        
        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = (t == num_threads - 1) ? count : (t + 1) * chunk_size;
            
            threads.emplace_back([=, &func]() {
                for (size_t i = start; i < end; ++i) {
                    func(i);
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    bool is_available() const override { return cuda_available_ || opencl_available_; }
    HardwareTarget get_target() const override { 
        return cuda_available_ ? HardwareTarget::GPU_CUDA : HardwareTarget::GPU_OpenCL; 
    }
    size_t get_max_concurrency() const override;
    
private:
    void check_gpu_availability();
};

/**
 * @brief FPGA-based execution strategy (HLS)
 */
class FPGAStrategy : public ExecutionStrategy {
private:
    ParallelConfig config_;
    bool fpga_available_;
    
public:
    explicit FPGAStrategy(const ParallelConfig& config);
    
    std::future<void> execute_void(std::function<void()> func) override {
        // For now, fallback to CPU execution with a warning
        std::cerr << "Warning: FPGA execution not fully implemented, falling back to CPU\n";
        
        auto promise = std::make_shared<std::promise<void>>();
        auto future = promise->get_future();
        
        std::thread([=, func = std::move(func)]() mutable {
            try {
                func();
                promise->set_value();
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        }).detach();
        
        return future;
    }
    
    void bulk_execute_void(size_t count, std::function<void(size_t)> func) override {
        // For now, fallback to sequential execution
        std::cerr << "Warning: FPGA bulk execution not fully implemented, falling back to sequential\n";
        
        for (size_t i = 0; i < count; ++i) {
            func(i);
        }
    }
    
    bool is_available() const override { return fpga_available_; }
    HardwareTarget get_target() const override { return HardwareTarget::FPGA_HLS; }
    size_t get_max_concurrency() const override { return 1; } // Typically single FPGA device
    
private:
    void check_fpga_availability();
};

/**
 * @brief Factory for creating execution strategies
 */
class ExecutionStrategyFactory {
public:
    static std::unique_ptr<ExecutionStrategy> create(const ParallelConfig& config);
    static std::unique_ptr<ExecutionStrategy> create_auto(const ParallelConfig& base_config);
    static std::vector<HardwareTarget> get_available_targets();
    static bool is_target_available(HardwareTarget target);
};

/**
 * @brief Main parallelism facade providing unified interface
 * 
 * This facade implements the Facade pattern to hide the complexity of different
 * parallel execution strategies and provides a simple, unified interface for users.
 */
class ParallelismFacade {
private:
    std::unique_ptr<ExecutionStrategy> strategy_;
    ParallelConfig config_;
    
public:
    explicit ParallelismFacade(const ParallelConfig& config = {});
    
    // Configuration methods
    void set_hardware_target(HardwareTarget target);
    void set_parallel_paradigm(ParallelParadigm paradigm);
    void set_max_workers(size_t workers);
    void configure(const ParallelConfig& config);
    
    // Execution methods
    template<typename F, typename... Args>
    auto execute(F&& func, Args&&... args) 
        -> std::future<std::invoke_result_t<F, Args...>> {
        return strategy_->execute(std::forward<F>(func), std::forward<Args>(args)...);
    }
    
    template<typename Iterator, typename F>
    void parallel_for_each(Iterator first, Iterator last, F&& func) {
        strategy_->bulk_execute(first, last, std::forward<F>(func));
    }
    
    template<typename Container, typename F>
    void parallel_for_each(Container& container, F&& func) {
        parallel_for_each(container.begin(), container.end(), std::forward<F>(func));
    }
    
    // Modern C++ async support
    template<typename F, typename... Args>
    auto async(F&& func, Args&&... args) 
        -> std::future<std::invoke_result_t<F, Args...>> {
        return execute(std::forward<F>(func), std::forward<Args>(args)...);
    }
    
    // Information methods
    HardwareTarget get_current_target() const;
    size_t get_max_concurrency() const;
    bool is_target_available(HardwareTarget target) const;
    std::vector<HardwareTarget> get_available_targets() const;
    
    // Performance monitoring
    struct PerformanceMetrics {
        size_t tasks_executed = 0;
        double avg_execution_time_ms = 0.0;
        double throughput_tasks_per_sec = 0.0;
        size_t active_workers = 0;
    };
    
    PerformanceMetrics get_performance_metrics() const;
};

/**
 * @brief Global parallelism facade instance
 */
inline ParallelismFacade& global_parallelism() {
    static ParallelismFacade instance;
    return instance;
}

/**
 * @brief Convenience functions for common parallel patterns
 */
namespace patterns {

// Parallel map operation
template<typename Iterator, typename F>
auto parallel_map(Iterator first, Iterator last, F&& func) 
    -> std::vector<std::invoke_result_t<F, typename Iterator::value_type>> {
    
    using result_type = std::invoke_result_t<F, typename Iterator::value_type>;
    std::vector<result_type> results;
    results.reserve(std::distance(first, last));
    
    std::vector<std::future<result_type>> futures;
    for (auto it = first; it != last; ++it) {
        futures.emplace_back(global_parallelism().async(func, *it));
    }
    
    for (auto& future : futures) {
        results.emplace_back(future.get());
    }
    
    return results;
}

// Parallel reduce operation
template<typename Iterator, typename T, typename BinaryOp>
T parallel_reduce(Iterator first, Iterator last, T init, BinaryOp&& op) {
    const size_t num_threads = global_parallelism().get_max_concurrency();
    const size_t length = std::distance(first, last);
    
    if (length < num_threads * 2) {
        // Small range, use sequential reduction
        return std::reduce(first, last, init, op);
    }
    
    const size_t chunk_size = length / num_threads;
    std::vector<std::future<T>> futures;
    
    auto chunk_start = first;
    for (size_t i = 0; i < num_threads - 1; ++i) {
        auto chunk_end = std::next(chunk_start, chunk_size);
        futures.emplace_back(global_parallelism().async([=, &op]() {
            return std::reduce(chunk_start, chunk_end, T{}, op);
        }));
        chunk_start = chunk_end;
    }
    
    // Last chunk handles remaining elements
    futures.emplace_back(global_parallelism().async([=, &op]() {
        return std::reduce(chunk_start, last, T{}, op);
    }));
    
    // Combine results
    T result = init;
    for (auto& future : futures) {
        result = op(result, future.get());
    }
    
    return result;
}

} // namespace patterns

} // namespace diffeq::execution