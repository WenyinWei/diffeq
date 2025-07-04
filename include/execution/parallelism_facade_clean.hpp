#pragma once

#include <memory>
#include <future>
#include <vector>
#include <functional>
#include <type_traits>
#include <thread>
#include <iostream>
#include <algorithm>

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
 * @brief Task priority levels
 */
enum class Priority : uint8_t {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
    Realtime = 4
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
    ParallelConfig config_;
    
public:
    explicit CPUStrategy(const ParallelConfig& config) : config_(config) {}
    
    std::future<void> execute_void(std::function<void()> func) override {
        auto task = std::make_shared<std::packaged_task<void()>>(std::move(func));
        auto future = task->get_future();
        
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
    explicit GPUStrategy(const ParallelConfig& config) 
        : config_(config), cuda_available_(false), opencl_available_(false) {
        check_gpu_availability();
    }
    
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
    size_t get_max_concurrency() const override {
        return cuda_available_ ? 2048 : (opencl_available_ ? 1024 : 1);
    }
    
private:
    void check_gpu_availability() {
        // In a full implementation, this would check for CUDA/OpenCL runtime
        cuda_available_ = false;
        opencl_available_ = false;
    }
};

/**
 * @brief FPGA-based execution strategy (HLS)
 */
class FPGAStrategy : public ExecutionStrategy {
private:
    ParallelConfig config_;
    bool fpga_available_;
    
public:
    explicit FPGAStrategy(const ParallelConfig& config) 
        : config_(config), fpga_available_(false) {
        check_fpga_availability();
    }
    
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
    size_t get_max_concurrency() const override { return 1; }
    
private:
    void check_fpga_availability() {
        fpga_available_ = false;
    }
};

/**
 * @brief Factory for creating execution strategies
 */
class ExecutionStrategyFactory {
public:
    static std::unique_ptr<ExecutionStrategy> create(const ParallelConfig& config) {
        switch (config.target) {
            case HardwareTarget::CPU_Sequential:
            case HardwareTarget::CPU_OpenMP:
            case HardwareTarget::CPU_MPI:
            case HardwareTarget::CPU_ThreadPool:
                return std::make_unique<CPUStrategy>(config);
                
            case HardwareTarget::GPU_CUDA:
            case HardwareTarget::GPU_OpenCL:
                return std::make_unique<GPUStrategy>(config);
                
            case HardwareTarget::FPGA_HLS:
                return std::make_unique<FPGAStrategy>(config);
                
            case HardwareTarget::Auto:
                return create_auto(config);
                
            default:
                return std::make_unique<CPUStrategy>(config);
        }
    }
    
    static std::unique_ptr<ExecutionStrategy> create_auto(const ParallelConfig& base_config) {
        // Try GPU first (highest performance potential)
        auto gpu_config = base_config;
        gpu_config.target = HardwareTarget::GPU_CUDA;
        auto gpu_strategy = std::make_unique<GPUStrategy>(gpu_config);
        if (gpu_strategy->is_available()) {
            return gpu_strategy;
        }
        
        // Try OpenCL GPU
        gpu_config.target = HardwareTarget::GPU_OpenCL;
        gpu_strategy = std::make_unique<GPUStrategy>(gpu_config);
        if (gpu_strategy->is_available()) {
            return gpu_strategy;
        }
        
        // Try FPGA
        auto fpga_config = base_config;
        fpga_config.target = HardwareTarget::FPGA_HLS;
        auto fpga_strategy = std::make_unique<FPGAStrategy>(fpga_config);
        if (fpga_strategy->is_available()) {
            return fpga_strategy;
        }
        
        // Fallback to CPU
        auto cpu_config = base_config;
        cpu_config.target = HardwareTarget::CPU_ThreadPool;
        return std::make_unique<CPUStrategy>(cpu_config);
    }
    
    static std::vector<HardwareTarget> get_available_targets() {
        std::vector<HardwareTarget> targets;
        
        // CPU is always available
        targets.push_back(HardwareTarget::CPU_Sequential);
        targets.push_back(HardwareTarget::CPU_ThreadPool);
        
        // Check GPU availability
        GPUStrategy gpu_test({});
        if (gpu_test.is_available()) {
            targets.push_back(HardwareTarget::GPU_CUDA);
            targets.push_back(HardwareTarget::GPU_OpenCL);
        }
        
        // Check FPGA availability
        FPGAStrategy fpga_test({});
        if (fpga_test.is_available()) {
            targets.push_back(HardwareTarget::FPGA_HLS);
        }
        
        return targets;
    }
    
    static bool is_target_available(HardwareTarget target) {
        auto available = get_available_targets();
        return std::find(available.begin(), available.end(), target) != available.end();
    }
};

/**
 * @brief Main parallelism facade providing unified interface
 */
class ParallelismFacade {
private:
    std::unique_ptr<ExecutionStrategy> strategy_;
    ParallelConfig config_;
    
public:
    explicit ParallelismFacade(const ParallelConfig& config = {}) 
        : config_(config) {
        strategy_ = ExecutionStrategyFactory::create(config_);
    }
    
    // Configuration methods
    void set_hardware_target(HardwareTarget target) {
        config_.target = target;
        strategy_ = ExecutionStrategyFactory::create(config_);
    }
    
    void set_parallel_paradigm(ParallelParadigm paradigm) {
        config_.paradigm = paradigm;
        strategy_ = ExecutionStrategyFactory::create(config_);
    }
    
    void set_max_workers(size_t workers) {
        config_.max_workers = workers;
        strategy_ = ExecutionStrategyFactory::create(config_);
    }
    
    void configure(const ParallelConfig& config) {
        config_ = config;
        strategy_ = ExecutionStrategyFactory::create(config_);
    }
    
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
    HardwareTarget get_current_target() const {
        return strategy_->get_target();
    }
    
    size_t get_max_concurrency() const {
        return strategy_->get_max_concurrency();
    }
    
    bool is_target_available(HardwareTarget target) const {
        return ExecutionStrategyFactory::is_target_available(target);
    }
    
    std::vector<HardwareTarget> get_available_targets() const {
        return ExecutionStrategyFactory::get_available_targets();
    }
    
    // Performance monitoring
    struct PerformanceMetrics {
        size_t tasks_executed = 0;
        double avg_execution_time_ms = 0.0;
        double throughput_tasks_per_sec = 0.0;
        size_t active_workers = 0;
    };
    
    PerformanceMetrics get_performance_metrics() const {
        PerformanceMetrics metrics;
        metrics.active_workers = get_max_concurrency();
        return metrics;
    }
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
        T result = init;
        for (auto it = first; it != last; ++it) {
            result = op(result, *it);
        }
        return result;
    }
    
    const size_t chunk_size = length / num_threads;
    std::vector<std::future<T>> futures;
    
    auto chunk_start = first;
    for (size_t i = 0; i < num_threads - 1; ++i) {
        auto chunk_end = std::next(chunk_start, chunk_size);
        futures.emplace_back(global_parallelism().async([=, &op]() {
            T result = T{};
            for (auto it = chunk_start; it != chunk_end; ++it) {
                result = op(result, *it);
            }
            return result;
        }));
        chunk_start = chunk_end;
    }
    
    // Last chunk handles remaining elements
    futures.emplace_back(global_parallelism().async([=, &op]() {
        T result = T{};
        for (auto it = chunk_start; it != last; ++it) {
            result = op(result, *it);
        }
        return result;
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