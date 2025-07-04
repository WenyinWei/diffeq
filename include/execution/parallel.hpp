#pragma once

#include <execution/parallelism_facade_clean.hpp>
#include <vector>
#include <functional>
#include <future>
#include <type_traits>

namespace diffeq::execution {

/**
 * @brief Simplified parallel execution interface for the diffeq library
 * 
 * This provides a much simpler interface for common parallel operations
 * without the complexity of the full parallelism facade system.
 * Users can access advanced features through the full ParallelismFacade if needed.
 */
class Parallel {
private:
    ParallelismFacade facade_;

public:
    /**
     * @brief Create a parallel executor with automatic hardware selection
     */
    Parallel() = default;

    /**
     * @brief Create a parallel executor with specified number of workers
     */
    explicit Parallel(size_t num_workers) {
        ParallelConfig config;
        config.max_workers = num_workers;
        config.target = HardwareTarget::Auto;
        facade_.configure(config);
    }

    /**
     * @brief Execute a function on each element in parallel
     */
    template<typename Iterator, typename F>
    void for_each(Iterator first, Iterator last, F&& func) {
        facade_.parallel_for_each(first, last, std::forward<F>(func));
    }

    /**
     * @brief Execute a function on each element in a container in parallel
     */
    template<typename Container, typename F>
    void for_each(Container& container, F&& func) {
        facade_.parallel_for_each(container.begin(), container.end(), std::forward<F>(func));
    }

    /**
     * @brief Execute a function asynchronously and return a future
     */
    template<typename F, typename... Args>
    auto async(F&& func, Args&&... args) 
        -> std::future<std::invoke_result_t<F, Args...>> {
        return facade_.async(std::forward<F>(func), std::forward<Args>(args)...);
    }

    /**
     * @brief Get the number of parallel workers available
     */
    size_t worker_count() const {
        return facade_.get_max_concurrency();
    }

    /**
     * @brief Check if GPU acceleration is available
     */
    bool gpu_available() const {
        return facade_.is_target_available(HardwareTarget::GPU_CUDA) ||
               facade_.is_target_available(HardwareTarget::GPU_OpenCL);
    }

    /**
     * @brief Enable GPU acceleration if available
     */
    void use_gpu() {
        if (gpu_available()) {
            facade_.set_hardware_target(HardwareTarget::GPU_CUDA);
        }
    }

    /**
     * @brief Force CPU-only execution
     */
    void use_cpu() {
        facade_.set_hardware_target(HardwareTarget::CPU_ThreadPool);
    }

    /**
     * @brief Set the number of worker threads
     */
    void set_workers(size_t count) {
        facade_.set_max_workers(count);
    }
};

/**
 * @brief Get the global parallel execution instance
 */
inline Parallel& parallel() {
    static Parallel instance;
    return instance;
}

/**
 * @brief Convenience function for parallel for_each operation
 */
template<typename Iterator, typename F>
void parallel_for_each(Iterator first, Iterator last, F&& func) {
    parallel().for_each(first, last, std::forward<F>(func));
}

/**
 * @brief Convenience function for parallel for_each on containers
 */
template<typename Container, typename F>
void parallel_for_each(Container& container, F&& func) {
    parallel().for_each(container, std::forward<F>(func));
}

/**
 * @brief Convenience function for async execution
 */
template<typename F, typename... Args>
auto parallel_async(F&& func, Args&&... args) 
    -> std::future<std::invoke_result_t<F, Args...>> {
    return parallel().async(std::forward<F>(func), std::forward<Args>(args)...);
}

/**
 * @brief Configure global parallel execution with number of workers
 */
inline void set_parallel_workers(size_t count) {
    parallel().set_workers(count);
}

/**
 * @brief Enable GPU acceleration globally if available
 */
inline void enable_gpu_acceleration() {
    parallel().use_gpu();
}

/**
 * @brief Force CPU-only execution globally
 */
inline void enable_cpu_only() {
    parallel().use_cpu();
}

} // namespace diffeq::execution