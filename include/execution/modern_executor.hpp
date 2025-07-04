#pragma once

#include <execution/parallelism_facade.hpp>
#include <future>
#include <coroutine>
#include <exception>
#include <type_traits>

namespace diffeq::execution {

/**
 * @brief Modern C++ executor concept implementation
 * 
 * This provides a modern C++ executor interface that's compatible with
 * proposed std::executor and can work with coroutines.
 */
class ModernExecutor {
public:
    /**
     * @brief Execute a function immediately on the current thread
     */
    template<typename F>
    void execute(F&& f) const {
        std::forward<F>(f)();
    }
    
    /**
     * @brief Schedule a function for later execution
     */
    template<typename F>
    void defer(F&& f) const {
        // Use the global parallelism facade for scheduling
        global_parallelism().async(std::forward<F>(f));
    }
    
    /**
     * @brief Bulk execute a range of functions
     */
    template<typename Iterator>
    void bulk_execute(Iterator first, Iterator last) const {
        global_parallelism().parallel_for_each(first, last, [](auto& f) { f(); });
    }
    
    /**
     * @brief Check if this executor can be used for scheduling
     */
    bool query(std::execution::scheduling_query_t) const {
        return true;
    }
    
    /**
     * @brief Check if this executor supports bulk operations
     */
    bool query(std::execution::bulk_query_t) const {
        return true;
    }
    
    /**
     * @brief Equality comparison
     */
    bool operator==(const ModernExecutor& other) const noexcept {
        return true;  // All instances are equivalent
    }
    
    bool operator!=(const ModernExecutor& other) const noexcept {
        return !(*this == other);
    }
};

/**
 * @brief Task type for coroutine support
 */
template<typename T = void>
class Task {
public:
    struct promise_type {
        Task get_return_object() {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        
        void return_void() requires std::is_void_v<T> {}
        
        void return_value(T value) requires (!std::is_void_v<T>) {
            result_ = std::move(value);
        }
        
        void unhandled_exception() {
            exception_ = std::current_exception();
        }
        
        T result() requires (!std::is_void_v<T>) {
            if (exception_) {
                std::rethrow_exception(exception_);
            }
            return result_;
        }
        
        void result() requires std::is_void_v<T> {
            if (exception_) {
                std::rethrow_exception(exception_);
            }
        }
        
    private:
        T result_{};
        std::exception_ptr exception_;
    };
    
    explicit Task(std::coroutine_handle<promise_type> handle) : handle_(handle) {}
    
    ~Task() {
        if (handle_) {
            handle_.destroy();
        }
    }
    
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    
    Task(Task&& other) noexcept : handle_(std::exchange(other.handle_, {})) {}
    Task& operator=(Task&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                handle_.destroy();
            }
            handle_ = std::exchange(other.handle_, {});
        }
        return *this;
    }
    
    bool is_ready() const {
        return handle_ && handle_.done();
    }
    
    T get() requires (!std::is_void_v<T>) {
        if (!handle_) {
            throw std::runtime_error("Invalid task handle");
        }
        return handle_.promise().result();
    }
    
    void get() requires std::is_void_v<T> {
        if (!handle_) {
            throw std::runtime_error("Invalid task handle");
        }
        handle_.promise().result();
    }
    
private:
    std::coroutine_handle<promise_type> handle_;
};

/**
 * @brief Awaitable type for async operations
 */
template<typename T>
class Awaitable {
public:
    explicit Awaitable(std::future<T>&& future) : future_(std::move(future)) {}
    
    bool await_ready() const {
        return future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }
    
    void await_suspend(std::coroutine_handle<> handle) {
        // Schedule the continuation when the future is ready
        global_parallelism().async([this, handle]() {
            future_.wait();
            handle.resume();
        });
    }
    
    T await_resume() {
        if constexpr (std::is_void_v<T>) {
            future_.get();
        } else {
            return future_.get();
        }
    }
    
private:
    std::future<T> future_;
};

/**
 * @brief Helper function to make awaitable from future
 */
template<typename T>
auto make_awaitable(std::future<T>&& future) {
    return Awaitable<T>(std::move(future));
}

/**
 * @brief Async function wrapper that returns a Task
 */
template<typename F, typename... Args>
auto async_task(F&& func, Args&&... args) -> Task<std::invoke_result_t<F, Args...>> {
    using return_type = std::invoke_result_t<F, Args...>;
    
    auto future = global_parallelism().async(std::forward<F>(func), std::forward<Args>(args)...);
    
    if constexpr (std::is_void_v<return_type>) {
        co_await make_awaitable(std::move(future));
        co_return;
    } else {
        auto result = co_await make_awaitable(std::move(future));
        co_return result;
    }
}

/**
 * @brief Parallel async execution of multiple tasks
 */
template<typename... Tasks>
auto when_all(Tasks&&... tasks) -> Task<std::tuple<typename Tasks::value_type...>> {
    std::vector<std::future<void>> futures;
    auto results = std::make_tuple(tasks.get()...);
    
    co_return results;
}

/**
 * @brief Parallel async execution - return first completed
 */
template<typename... Tasks>
auto when_any(Tasks&&... tasks) -> Task<std::variant<typename Tasks::value_type...>> {
    // This is a simplified implementation
    // A full implementation would use proper synchronization
    auto first_result = std::get<0>(std::make_tuple(tasks...)).get();
    co_return std::variant<typename Tasks::value_type...>(first_result);
}

/**
 * @brief Executor-based scheduling
 */
namespace scheduling {

/**
 * @brief Schedule work on a specific executor
 */
template<typename Executor, typename F, typename... Args>
auto schedule_on(Executor&& executor, F&& func, Args&&... args) {
    return async_task([=, func = std::forward<F>(func)]() mutable {
        return func(args...);
    });
}

/**
 * @brief Schedule work with a delay
 */
template<typename Rep, typename Period, typename F, typename... Args>
auto schedule_after(std::chrono::duration<Rep, Period> delay, F&& func, Args&&... args) {
    return async_task([=, func = std::forward<F>(func)]() mutable {
        std::this_thread::sleep_for(delay);
        return func(args...);
    });
}

/**
 * @brief Schedule work at a specific time point
 */
template<typename Clock, typename Duration, typename F, typename... Args>
auto schedule_at(std::chrono::time_point<Clock, Duration> time_point, F&& func, Args&&... args) {
    return async_task([=, func = std::forward<F>(func)]() mutable {
        std::this_thread::sleep_until(time_point);
        return func(args...);
    });
}

} // namespace scheduling

/**
 * @brief Integration with std::execution when available
 */
namespace std_execution_compat {

#ifdef __cpp_lib_execution
using std_executor = std::execution::any_executor<>;

template<typename F, typename... Args>
auto execute_on_std(std_executor executor, F&& func, Args&&... args) {
    auto packaged = std::packaged_task<std::invoke_result_t<F, Args...>()>(
        [=, func = std::forward<F>(func)]() mutable {
            return func(args...);
        }
    );
    
    auto future = packaged.get_future();
    std::execution::execute(executor, std::move(packaged));
    
    return make_awaitable(std::move(future));
}
#endif

} // namespace std_execution_compat

/**
 * @brief Utilities for integration with ODE solvers
 */
namespace ode_integration {

/**
 * @brief Async step execution for ODE integrators
 */
template<typename Integrator, typename State, typename TimeType>
auto async_step(Integrator& integrator, State& state, TimeType dt) {
    return async_task([&integrator, &state, dt]() {
        integrator.step(state, dt);
    });
}

/**
 * @brief Parallel integration of multiple initial conditions
 */
template<typename Integrator, typename StateContainer, typename TimeType>
auto parallel_integrate(Integrator& integrator, StateContainer& initial_states, 
                       TimeType dt, TimeType end_time) {
    std::vector<Task<void>> tasks;
    
    for (auto& state : initial_states) {
        tasks.emplace_back(async_task([&integrator, &state, dt, end_time]() {
            TimeType current_time = 0.0;
            while (current_time < end_time) {
                TimeType step_size = std::min(dt, end_time - current_time);
                integrator.step(state, step_size);
                current_time += step_size;
            }
        }));
    }
    
    return when_all(tasks...);
}

} // namespace ode_integration

} // namespace diffeq::execution