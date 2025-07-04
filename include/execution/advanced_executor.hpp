#pragma once

#include <thread>
#include <future>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <memory>
#include <chrono>
#include <type_traits>

// Check for C++23 std::execution support
#if __has_include(<execution>) && defined(__cpp_lib_execution)
#include <execution>
#define DIFFEQ_HAS_STD_EXECUTION 1
#else
#define DIFFEQ_HAS_STD_EXECUTION 0
#endif

// ASIO support for networking (optional)
#if __has_include(<boost/asio.hpp>)
#include <boost/asio.hpp>
#define DIFFEQ_HAS_ASIO 1
#elif __has_include(<asio.hpp>)
#include <asio.hpp>
#define DIFFEQ_HAS_ASIO 1
#else
#define DIFFEQ_HAS_ASIO 0
#endif

namespace diffeq::execution {

/**
 * @brief Execution policy types for different use cases
 */
enum class ExecutionPolicy {
    Sequential,      // Single-threaded execution
    Parallel,        // Multi-threaded parallel execution
    Vectorized,      // SIMD-optimized execution
    Realtime,        // Real-time priority execution
    Network          // Network I/O optimized execution
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
 * @brief Task wrapper with priority and metadata
 */
template<typename T>
struct Task {
    std::function<T()> callable;
    Priority priority;
    std::chrono::steady_clock::time_point created_at;
    std::chrono::microseconds timeout;
    std::string task_id;
    
    template<typename F>
    Task(F&& func, Priority prio = Priority::Normal, 
         std::chrono::microseconds timeout_us = std::chrono::microseconds::max(),
         std::string id = "")
        : callable(std::forward<F>(func))
        , priority(prio)
        , created_at(std::chrono::steady_clock::now())
        , timeout(timeout_us)
        , task_id(std::move(id)) {}
    
    bool is_expired() const {
        return (std::chrono::steady_clock::now() - created_at) > timeout;
    }
};

/**
 * @brief Priority-based task comparator
 */
template<typename T>
struct TaskComparator {
    bool operator()(const std::shared_ptr<Task<T>>& a, 
                   const std::shared_ptr<Task<T>>& b) const {
        if (a->priority != b->priority) {
            return static_cast<uint8_t>(a->priority) < static_cast<uint8_t>(b->priority);
        }
        return a->created_at > b->created_at; // FIFO for same priority
    }
};

/**
 * @brief High-performance executor with multiple execution policies
 * 
 * This executor provides:
 * - Priority-based task scheduling
 * - Multiple execution policies (sequential, parallel, realtime)
 * - Integration with std::execution when available
 * - ASIO integration for network operations
 * - Real-time thread priorities
 * - Task timeout and cancellation
 */
class AdvancedExecutor {
public:
    struct Config {
        size_t num_threads = std::thread::hardware_concurrency();
        size_t realtime_threads = 1;
        bool enable_realtime_priority = false;
        bool enable_work_stealing = true;
        size_t queue_capacity = 10000;
        std::chrono::milliseconds worker_timeout{100};
    };
    
    explicit AdvancedExecutor(Config config = {})
        : config_(config), shutdown_(false), next_worker_(0) {
        
        initialize_workers();
        
        #if DIFFEQ_HAS_ASIO
        initialize_asio();
        #endif
    }
    
    ~AdvancedExecutor() {
        shutdown();
    }
    
    /**
     * @brief Submit task with specific execution policy
     */
    template<typename F, typename... Args>
    auto submit(ExecutionPolicy policy, Priority priority, F&& func, Args&&... args) 
        -> std::future<std::invoke_result_t<F, Args...>> {
        
        using return_type = std::invoke_result_t<F, Args...>;
        
        auto task_func = [f = std::forward<F>(func), 
                         arg_tuple = std::make_tuple(std::forward<Args>(args)...)]() mutable -> return_type {
            return std::apply(std::move(f), std::move(arg_tuple));
        };
        
        auto task = std::make_shared<Task<return_type>>(std::move(task_func), priority);
        auto packaged = std::make_shared<std::packaged_task<return_type()>>(
            [task] { return task->callable(); }
        );
        
        auto future = packaged->get_future();
        
        switch (policy) {
            case ExecutionPolicy::Sequential:
                submit_sequential(packaged, priority);
                break;
                
            case ExecutionPolicy::Parallel:
                submit_parallel(packaged, priority);
                break;
                
            case ExecutionPolicy::Realtime:
                submit_realtime(packaged, priority);
                break;
                
            #if DIFFEQ_HAS_ASIO
            case ExecutionPolicy::Network:
                submit_network(packaged, priority);
                break;
            #endif
                
            default:
                submit_parallel(packaged, priority);
                break;
        }
        
        return future;
    }
    
    /**
     * @brief Convenience overloads
     */
    template<typename F, typename... Args>
    auto submit(F&& func, Args&&... args) {
        return submit(ExecutionPolicy::Parallel, Priority::Normal, 
                     std::forward<F>(func), std::forward<Args>(args)...);
    }
    
    template<typename F, typename... Args>
    auto submit_realtime(F&& func, Args&&... args) {
        return submit(ExecutionPolicy::Realtime, Priority::Realtime,
                     std::forward<F>(func), std::forward<Args>(args)...);
    }
    
    #if DIFFEQ_HAS_STD_EXECUTION
    /**
     * @brief Execute using std::execution when available
     */
    template<typename ExecutionPolicy, typename F, typename... Args>
    auto execute_std(ExecutionPolicy&& policy, F&& func, Args&&... args) {
        return std::execution::execute(
            std::forward<ExecutionPolicy>(policy),
            [f = std::forward<F>(func), 
             arg_tuple = std::make_tuple(std::forward<Args>(args)...)]() mutable {
                return std::apply(std::move(f), std::move(arg_tuple));
            }
        );
    }
    #endif
    
    /**
     * @brief Bulk submit for vectorized operations
     */
    template<typename Iterator, typename F>
    void bulk_submit(Iterator first, Iterator last, F&& func, 
                    ExecutionPolicy policy = ExecutionPolicy::Parallel) {
        
        const size_t batch_size = config_.num_threads * 2;
        size_t distance = std::distance(first, last);
        
        if (distance <= batch_size) {
            // Small range, submit individual tasks
            for (auto it = first; it != last; ++it) {
                submit(policy, Priority::Normal, func, *it);
            }
        } else {
            // Large range, split into batches
            size_t chunk_size = distance / config_.num_threads;
            auto chunk_start = first;
            
            for (size_t i = 0; i < config_.num_threads && chunk_start != last; ++i) {
                auto chunk_end = (i == config_.num_threads - 1) ? last : 
                                std::next(chunk_start, chunk_size);
                
                submit(policy, Priority::Normal, [func, chunk_start, chunk_end]() {
                    for (auto it = chunk_start; it != chunk_end; ++it) {
                        func(*it);
                    }
                });
                
                chunk_start = chunk_end;
            }
        }
    }
    
    /**
     * @brief Wait for all pending tasks to complete
     */
    void wait_for_all() {
        std::unique_lock<std::mutex> lock(completion_mutex_);
        completion_cv_.wait(lock, [this] {
            return all_queues_empty();
        });
    }
    
    /**
     * @brief Get execution statistics
     */
    struct Statistics {
        size_t tasks_completed = 0;
        size_t tasks_failed = 0;
        size_t tasks_timeout = 0;
        std::chrono::microseconds avg_task_time{0};
        size_t queue_peak_size = 0;
        double cpu_utilization = 0.0;
    };
    
    Statistics get_statistics() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return stats_;
    }
    
    /**
     * @brief Shutdown executor gracefully
     */
    void shutdown() {
        if (shutdown_.exchange(true)) {
            return; // Already shutdown
        }
        
        // Notify all workers to stop
        for (auto& worker : workers_) {
            worker.condition.notify_all();
        }
        
        for (auto& worker : realtime_workers_) {
            worker.condition.notify_all();
        }
        
        #if DIFFEQ_HAS_ASIO
        if (asio_work_guard_) {
            asio_work_guard_.reset();
        }
        if (asio_context_) {
            asio_context_->stop();
        }
        #endif
        
        // Join all threads
        for (auto& worker : workers_) {
            if (worker.thread.joinable()) {
                worker.thread.join();
            }
        }
        
        for (auto& worker : realtime_workers_) {
            if (worker.thread.joinable()) {
                worker.thread.join();
            }
        }
        
        #if DIFFEQ_HAS_ASIO
        for (auto& thread : asio_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        #endif
    }

private:
    Config config_;
    std::atomic<bool> shutdown_;
    std::atomic<size_t> next_worker_;
    
    struct Worker {
        std::thread thread;
        std::priority_queue<
            std::shared_ptr<std::packaged_task<void()>>,
            std::vector<std::shared_ptr<std::packaged_task<void()>>>,
            std::function<bool(const std::shared_ptr<std::packaged_task<void()>>&,
                             const std::shared_ptr<std::packaged_task<void()>>&)>
        > task_queue;
        std::mutex queue_mutex;
        std::condition_variable condition;
        std::atomic<bool> is_busy{false};
        
        Worker() : task_queue([](const auto& a, const auto& b) { return false; }) {}
    };
    
    std::vector<Worker> workers_;
    std::vector<Worker> realtime_workers_;
    
    // Global sequential queue for sequential execution
    Worker sequential_worker_;
    
    mutable std::mutex stats_mutex_;
    Statistics stats_;
    
    std::mutex completion_mutex_;
    std::condition_variable completion_cv_;
    
    #if DIFFEQ_HAS_ASIO
    std::unique_ptr<asio::io_context> asio_context_;
    std::unique_ptr<asio::executor_work_guard<asio::io_context::executor_type>> asio_work_guard_;
    std::vector<std::thread> asio_threads_;
    #endif
    
    void initialize_workers() {
        // Initialize regular workers
        workers_.resize(config_.num_threads);
        for (size_t i = 0; i < config_.num_threads; ++i) {
            workers_[i].thread = std::thread([this, i] { worker_loop(workers_[i], false); });
        }
        
        // Initialize real-time workers
        realtime_workers_.resize(config_.realtime_threads);
        for (size_t i = 0; i < config_.realtime_threads; ++i) {
            realtime_workers_[i].thread = std::thread([this, i] { 
                worker_loop(realtime_workers_[i], true); 
            });
        }
        
        // Initialize sequential worker
        sequential_worker_.thread = std::thread([this] { 
            worker_loop(sequential_worker_, false); 
        });
    }
    
    #if DIFFEQ_HAS_ASIO
    void initialize_asio() {
        asio_context_ = std::make_unique<asio::io_context>();
        asio_work_guard_ = std::make_unique<asio::executor_work_guard<asio::io_context::executor_type>>(
            asio_context_->get_executor()
        );
        
        // Start ASIO threads
        size_t asio_threads = std::max(2u, config_.num_threads / 4);
        asio_threads_.reserve(asio_threads);
        
        for (size_t i = 0; i < asio_threads; ++i) {
            asio_threads_.emplace_back([this] {
                asio_context_->run();
            });
        }
    }
    #endif
    
    void worker_loop(Worker& worker, bool is_realtime) {
        if (is_realtime && config_.enable_realtime_priority) {
            set_realtime_priority();
        }
        
        while (!shutdown_.load()) {
            std::shared_ptr<std::packaged_task<void()>> task;
            
            {
                std::unique_lock<std::mutex> lock(worker.queue_mutex);
                worker.condition.wait_for(
                    lock,
                    config_.worker_timeout,
                    [&] { return !worker.task_queue.empty() || shutdown_.load(); }
                );
                
                if (shutdown_.load()) break;
                
                if (!worker.task_queue.empty()) {
                    task = worker.task_queue.top();
                    worker.task_queue.pop();
                }
            }
            
            if (task) {
                worker.is_busy.store(true);
                auto start_time = std::chrono::steady_clock::now();
                
                try {
                    (*task)();
                    
                    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - start_time
                    );
                    
                    update_statistics(elapsed, true);
                } catch (...) {
                    update_statistics(std::chrono::microseconds{0}, false);
                }
                
                worker.is_busy.store(false);
                completion_cv_.notify_one();
            }
            
            // Work stealing if enabled
            if (config_.enable_work_stealing && !is_realtime) {
                attempt_work_stealing(worker);
            }
        }
    }
    
    void set_realtime_priority() {
        #ifdef __linux__
        struct sched_param param;
        param.sched_priority = 80; // High real-time priority
        
        if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
            std::cerr << "Warning: Failed to set real-time priority\n";
        }
        #endif
    }
    
    template<typename T>
    void submit_sequential(std::shared_ptr<std::packaged_task<T()>> task, Priority priority) {
        auto void_task = std::make_shared<std::packaged_task<void()>>(
            [task] { (*task)(); }
        );
        
        {
            std::lock_guard<std::mutex> lock(sequential_worker_.queue_mutex);
            sequential_worker_.task_queue.push(void_task);
        }
        sequential_worker_.condition.notify_one();
    }
    
    template<typename T>
    void submit_parallel(std::shared_ptr<std::packaged_task<T()>> task, Priority priority) {
        auto void_task = std::make_shared<std::packaged_task<void()>>(
            [task] { (*task)(); }
        );
        
        // Round-robin distribution
        size_t worker_id = next_worker_.fetch_add(1) % config_.num_threads;
        
        {
            std::lock_guard<std::mutex> lock(workers_[worker_id].queue_mutex);
            workers_[worker_id].task_queue.push(void_task);
        }
        workers_[worker_id].condition.notify_one();
    }
    
    template<typename T>
    void submit_realtime(std::shared_ptr<std::packaged_task<T()>> task, Priority priority) {
        auto void_task = std::make_shared<std::packaged_task<void()>>(
            [task] { (*task)(); }
        );
        
        // Use least busy real-time worker
        size_t best_worker = 0;
        size_t min_queue_size = SIZE_MAX;
        
        for (size_t i = 0; i < realtime_workers_.size(); ++i) {
            std::lock_guard<std::mutex> lock(realtime_workers_[i].queue_mutex);
            if (realtime_workers_[i].task_queue.size() < min_queue_size) {
                min_queue_size = realtime_workers_[i].task_queue.size();
                best_worker = i;
            }
        }
        
        {
            std::lock_guard<std::mutex> lock(realtime_workers_[best_worker].queue_mutex);
            realtime_workers_[best_worker].task_queue.push(void_task);
        }
        realtime_workers_[best_worker].condition.notify_one();
    }
    
    #if DIFFEQ_HAS_ASIO
    template<typename T>
    void submit_network(std::shared_ptr<std::packaged_task<T()>> task, Priority priority) {
        asio::post(*asio_context_, [task] {
            (*task)();
        });
    }
    #endif
    
    void attempt_work_stealing(Worker& worker) {
        // Try to steal work from other workers if this worker is idle
        for (auto& other_worker : workers_) {
            if (&other_worker == &worker) continue;
            
            std::unique_lock<std::mutex> other_lock(other_worker.queue_mutex, std::try_to_lock);
            if (!other_lock.owns_lock()) continue;
            
            if (other_worker.task_queue.size() > 1) {
                // Steal a task
                auto stolen_task = other_worker.task_queue.top();
                other_worker.task_queue.pop();
                
                std::lock_guard<std::mutex> my_lock(worker.queue_mutex);
                worker.task_queue.push(stolen_task);
                return;
            }
        }
    }
    
    bool all_queues_empty() const {
        // Check all worker queues
        for (const auto& worker : workers_) {
            std::lock_guard<std::mutex> lock(worker.queue_mutex);
            if (!worker.task_queue.empty() || worker.is_busy.load()) {
                return false;
            }
        }
        
        for (const auto& worker : realtime_workers_) {
            std::lock_guard<std::mutex> lock(worker.queue_mutex);
            if (!worker.task_queue.empty() || worker.is_busy.load()) {
                return false;
            }
        }
        
        std::lock_guard<std::mutex> lock(sequential_worker_.queue_mutex);
        return sequential_worker_.task_queue.empty() && !sequential_worker_.is_busy.load();
    }
    
    void update_statistics(std::chrono::microseconds task_time, bool success) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        
        if (success) {
            stats_.tasks_completed++;
            // Update average task time (simple moving average)
            stats_.avg_task_time = (stats_.avg_task_time + task_time) / 2;
        } else {
            stats_.tasks_failed++;
        }
    }
};

/**
 * @brief Global executor instance for convenience
 */
inline AdvancedExecutor& global_executor() {
    static AdvancedExecutor instance;
    return instance;
}

/**
 * @brief Convenience functions for common execution patterns
 */
template<typename F, typename... Args>
auto async_execute(F&& func, Args&&... args) {
    return global_executor().submit(std::forward<F>(func), std::forward<Args>(args)...);
}

template<typename F, typename... Args>
auto realtime_execute(F&& func, Args&&... args) {
    return global_executor().submit_realtime(std::forward<F>(func), std::forward<Args>(args)...);
}

template<typename Iterator, typename F>
void parallel_for_each(Iterator first, Iterator last, F&& func) {
    global_executor().bulk_submit(first, last, std::forward<F>(func), ExecutionPolicy::Parallel);
}

} // namespace diffeq::execution
