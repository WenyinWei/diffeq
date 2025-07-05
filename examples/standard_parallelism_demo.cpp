#include <diffeq.hpp>
#include <vector>
#include <algorithm>
#include <execution>
#include <numeric>
#include <iostream>
#include <chrono>
#include <future>
#include <queue>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <memory>

// Conditional includes for optional libraries
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TBB_AVAILABLE
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/task_scheduler_init.h>
#endif

#ifdef THRUST_AVAILABLE
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#ifdef OPENCL_AVAILABLE
#include <CL/cl.hpp>
#endif

namespace diffeq::examples {

/**
 * @brief Example: Parallel ODE integration using std::execution policies
 * 
 * This shows how to use standard C++17/20 execution policies with
 * existing diffeq integrators - no custom parallel classes needed!
 */
template<typename State, typename Time>
class ODEStdExecution {
public:
    
    /**
     * @brief Integrate multiple initial conditions in parallel using std::execution
     * 
     * @param system The ODE system function
     * @param initial_conditions Vector of initial states
     * @param dt Time step
     * @param steps Number of integration steps
     */
    template<typename System>
    static void integrate_multiple_conditions(
        System&& system,
        std::vector<State>& initial_conditions,
        Time dt,
        int steps
    ) {
        // Use standard C++17 parallel execution - no custom classes!
        std::for_each(std::execution::par_unseq, 
                     initial_conditions.begin(), 
                     initial_conditions.end(),
                     [&](State& state) {
                         auto integrator = diffeq::integrators::ode::RK4Integrator<State, Time>(system);
                         for (int i = 0; i < steps; ++i) {
                             integrator.step(state, dt);
                         }
                     });
    }
    
    /**
     * @brief Parameter sweep using std::execution policies
     * 
     * Demonstrate flexibility beyond just initial conditions -
     * vary parameters, integrators, callbacks, etc.
     */
    template<typename System>
    static void parameter_sweep(
        System&& system_template,
        const State& initial_state,
        const std::vector<Time>& parameters,
        std::vector<State>& results,
        Time dt,
        int steps
    ) {
        results.resize(parameters.size());
        
        // Create indices for parallel processing
        std::vector<size_t> indices(parameters.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        // Parallel parameter sweep using standard library
        std::for_each(std::execution::par, 
                     indices.begin(), 
                     indices.end(),
                     [&](size_t i) {
                         State state = initial_state;
                         Time param = parameters[i];
                         
                         // Create system with this parameter
                         auto system = [&](Time t, const State& y, State& dydt) {
                             system_template(t, y, dydt, param);  // Pass parameter
                         };
                         
                         auto integrator = diffeq::integrators::ode::RK4Integrator<State, Time>(system);
                         for (int step = 0; step < steps; ++step) {
                             integrator.step(state, dt);
                         }
                         results[i] = state;
                     });
    }
};

#ifdef _OPENMP
/**
 * @brief OpenMP examples for CPU parallelism
 * 
 * OpenMP is widely supported and doesn't require custom infrastructure.
 */
template<typename State, typename Time>
class ODEOpenMP {
public:
    
    template<typename System>
    static void integrate_openmp(
        System&& system,
        std::vector<State>& states,
        Time dt,
        int steps
    ) {
        // Simple OpenMP parallel loop - no custom classes needed!
        #pragma omp parallel for
        for (size_t i = 0; i < states.size(); ++i) {
            auto integrator = diffeq::integrators::ode::RK4Integrator<State, Time>(system);
            for (int step = 0; step < steps; ++step) {
                integrator.step(states[i], dt);
            }
        }
    }
    
    /**
     * @brief Different integrators in parallel
     * Demonstrate running different algorithms simultaneously
     */
    template<typename System>
    static void multi_integrator_comparison(
        System&& system,
        const State& initial_state,
        Time dt,
        int steps,
        std::vector<State>& rk4_results,
        std::vector<State>& euler_results
    ) {
        const int num_runs = 100;
        rk4_results.resize(num_runs, initial_state);
        euler_results.resize(num_runs, initial_state);
        
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // RK4 integration
                #pragma omp parallel for
                for (int i = 0; i < num_runs; ++i) {
                    auto integrator = diffeq::integrators::ode::RK4Integrator<State, Time>(system);
                    for (int step = 0; step < steps; ++step) {
                        integrator.step(rk4_results[i], dt);
                    }
                }
            }
            
            #pragma omp section
            {
                // Euler integration (for comparison)
                #pragma omp parallel for
                for (int i = 0; i < num_runs; ++i) {
                    auto integrator = diffeq::integrators::ode::EulerIntegrator<State, Time>(system);
                    for (int step = 0; step < steps; ++step) {
                        integrator.step(euler_results[i], dt);
                    }
                }
            }
        }
    }
};
#endif

#ifdef TBB_AVAILABLE
/**
 * @brief Intel TBB examples for advanced parallelism
 */
template<typename State, typename Time>
class ODETBB {
public:
    
    template<typename System>
    static void integrate_tbb(
        System&& system,
        std::vector<State>& states,
        Time dt,
        int steps
    ) {
        // TBB parallel for - handles load balancing automatically
        tbb::parallel_for(tbb::blocked_range<size_t>(0, states.size()),
                         [&](const tbb::blocked_range<size_t>& range) {
                             for (size_t i = range.begin(); i != range.end(); ++i) {
                                 auto integrator = diffeq::integrators::ode::RK4Integrator<State, Time>(system);
                                 for (int step = 0; step < steps; ++step) {
                                     integrator.step(states[i], dt);
                                 }
                             }
                         });
    }
    
    template<typename System>
    static void integrate_blocked(
        System&& system,
        std::vector<State>& states,
        Time dt,
        int steps,
        size_t block_size = 1000
    ) {
        // TBB with custom block sizes for cache optimization
        tbb::parallel_for(tbb::blocked_range<size_t>(0, states.size(), block_size),
                         [&](const tbb::blocked_range<size_t>& range) {
                             for (size_t i = range.begin(); i != range.end(); ++i) {
                                 auto integrator = diffeq::integrators::ode::RK4Integrator<State, Time>(system);
                                 for (int step = 0; step < steps; ++step) {
                                     integrator.step(states[i], dt);
                                 }
                             }
                         });
    }
};
#endif

/**
 * @brief Task-based async dispatcher for ODE integration
 */
template<typename State, typename Time>
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

    template<typename System>
    std::future<State> submit_ode_task(
        System&& system,
        State initial_state,
        Time dt,
        int steps
    ) {
        auto promise = std::make_shared<std::promise<State>>();
        auto future = promise->get_future();
        
        auto task = [system = std::forward<System>(system),
                    initial_state,
                    dt,
                    steps,
                    promise]() mutable {
            State state = initial_state;
            auto integrator = diffeq::integrators::ode::RK4Integrator<State, Time>(system);
            
            for (int i = 0; i < steps; ++i) {
                integrator.step(state, dt);
            }
            
            promise->set_value(state);
        };
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_queue.push(std::move(task));
        }
        cv.notify_one();
        
        return future;
    }

    void start_async_processing(size_t num_threads = std::thread::hardware_concurrency()) {
        for (size_t i = 0; i < num_threads; ++i) {
            worker_threads.emplace_back([this]() {
                while (!stop_flag) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        cv.wait(lock, [this]() { return !task_queue.empty() || stop_flag; });
                        
                        if (stop_flag && task_queue.empty()) break;
                        
                        if (!task_queue.empty()) {
                            task = std::move(task_queue.front());
                            task_queue.pop();
                        }
                    }
                    
                    if (task) {
                        std::thread worker_thread(task);
                        worker_thread.detach();
                    }
                }
            });
        }
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

/**
 * @brief Unified parallel interface that automatically chooses the best backend
 */
template<typename State, typename Time>
class ODEParallel {
public:
    enum class Backend {
        Auto,           // Automatically choose best available
        StdExecution,   // C++17/20 std::execution
        OpenMP,         // OpenMP parallel loops
        TBB,            // Intel Threading Building Blocks
        Thrust,         // NVIDIA Thrust (GPU without kernels)
        Cuda,           // Direct CUDA kernels
        OpenCL          // OpenCL cross-platform
    };

private:
    Backend selected_backend = Backend::Auto;

    Backend detect_best_backend() const {
        // Simple detection logic - in practice would be more sophisticated
        #ifdef _OPENMP
        return Backend::OpenMP;
        #elif defined(TBB_AVAILABLE)
        return Backend::TBB;
        #elif defined(THRUST_AVAILABLE)
        return Backend::Thrust;
        #else
        return Backend::StdExecution;
        #endif
    }

public:
    ODEParallel() : selected_backend(Backend::Auto) {}
    
    explicit ODEParallel(Backend backend) : selected_backend(backend) {}
    
    void set_backend(Backend backend) {
        selected_backend = backend;
    }
    
    Backend get_backend() const {
        return selected_backend;
    }

    template<typename System>
    void integrate_parallel(
        System&& system,
        std::vector<State>& states,
        Time dt,
        int steps
    ) {
        Backend backend = (selected_backend == Backend::Auto) ? detect_best_backend() : selected_backend;
        
        switch (backend) {
            case Backend::StdExecution:
                ODEStdExecution<State, Time>::integrate_multiple_conditions(
                    std::forward<System>(system), states, dt, steps);
                break;
                
            #ifdef _OPENMP
            case Backend::OpenMP:
                ODEOpenMP<State, Time>::integrate_openmp(
                    std::forward<System>(system), states, dt, steps);
                break;
            #endif
                
            #ifdef TBB_AVAILABLE
            case Backend::TBB:
                ODETBB<State, Time>::integrate_tbb(
                    std::forward<System>(system), states, dt, steps);
                break;
            #endif
                
            default:
                // Fallback to std::execution
                ODEStdExecution<State, Time>::integrate_multiple_conditions(
                    std::forward<System>(system), states, dt, steps);
                break;
        }
    }

    template<typename System>
    void parameter_sweep(
        System&& system_template,
        const State& initial_state,
        const std::vector<Time>& parameters,
        std::vector<State>& results,
        Time dt,
        int steps
    ) {
        ODEStdExecution<State, Time>::parameter_sweep(
            std::forward<System>(system_template),
            initial_state, parameters, results, dt, steps);
    }

    static std::vector<std::pair<Backend, std::string>> available_backends() {
        std::vector<std::pair<Backend, std::string>> backends;
        
        backends.emplace_back(Backend::StdExecution, "C++17/20 std::execution");
        
        #ifdef _OPENMP
        backends.emplace_back(Backend::OpenMP, "OpenMP");
        #endif
        
        #ifdef TBB_AVAILABLE
        backends.emplace_back(Backend::TBB, "Intel TBB");
        #endif
        
        #ifdef THRUST_AVAILABLE
        backends.emplace_back(Backend::Thrust, "NVIDIA Thrust");
        #endif
        
        return backends;
    }

    static void print_available_backends() {
        auto backends = available_backends();
        std::cout << "Available parallel backends:" << std::endl;
        for (const auto& [backend, name] : backends) {
            std::cout << "  - " << name << std::endl;
        }
    }
};

// Convenience functions
template<typename System, typename State, typename Time>
void integrate_parallel(
    System&& system,
    std::vector<State>& states,
    Time dt,
    int steps
) {
    ODEParallel<State, Time> parallel;
    parallel.integrate_parallel(std::forward<System>(system), states, dt, steps);
}

template<typename System, typename State, typename Time>
void parameter_sweep_parallel(
    System&& system_template,
    const State& initial_state,
    const std::vector<Time>& parameters,
    std::vector<State>& results,
    Time dt,
    int steps
) {
    ODEParallel<State, Time> parallel;
    parallel.parameter_sweep(std::forward<System>(system_template),
                           initial_state, parameters, results, dt, steps);
}

template<typename State, typename Time>
std::unique_ptr<diffeq::examples::ODETaskDispatcher<State, Time>> create_async_dispatcher() {
    return std::make_unique<diffeq::examples::ODETaskDispatcher<State, Time>>();
}

} // namespace diffeq::examples

int main() {
    std::cout << "=== diffeq Standard Parallelism Examples ===" << std::endl;
    
    // Show available backends
    diffeq::examples::ODEParallel<std::vector<double>, double>::print_available_backends();
    
    // Define a simple ODE system: exponential decay
    auto system = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        dydt[0] = -0.1 * y[0];
        dydt[1] = -0.2 * y[1];
    };
    
    // Parameterized system for parameter sweep
    auto parameterized_system = [](double t, const std::vector<double>& y, std::vector<double>& dydt, double decay_rate) {
        dydt[0] = -decay_rate * y[0];
        dydt[1] = -decay_rate * 2.0 * y[1];
    };
    
    const int num_conditions = 1000;
    const double dt = 0.01;
    const int steps = 100;
    
    // Create initial conditions
    std::vector<std::vector<double>> initial_conditions;
    for (int i = 0; i < num_conditions; ++i) {
        initial_conditions.push_back({static_cast<double>(i), static_cast<double>(i) * 0.5});
    }
    
    std::cout << "\n=== Testing std::execution Parallelism ===" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Test std::execution parallelism
    diffeq::examples::ODEStdExecution<std::vector<double>, double>::integrate_multiple_conditions(
        system, initial_conditions, dt, steps);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "std::execution completed in " << duration.count() << "ms" << std::endl;
    std::cout << "Result for condition 10: [" << initial_conditions[10][0] << ", " << initial_conditions[10][1] << "]" << std::endl;
    
    #ifdef _OPENMP
    std::cout << "\n=== Testing OpenMP Parallelism ===" << std::endl;
    
    // Reset initial conditions
    for (int i = 0; i < num_conditions; ++i) {
        initial_conditions[i] = {static_cast<double>(i), static_cast<double>(i) * 0.5};
    }
    
    start_time = std::chrono::high_resolution_clock::now();
    
    diffeq::examples::ODEOpenMP<std::vector<double>, double>::integrate_openmp(
        system, initial_conditions, dt, steps);
    
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "OpenMP completed in " << duration.count() << "ms" << std::endl;
    std::cout << "Result for condition 10: [" << initial_conditions[10][0] << ", " << initial_conditions[10][1] << "]" << std::endl;
    #endif
    
    std::cout << "\n=== Testing Parameter Sweep ===" << std::endl;
    
    std::vector<double> decay_rates = {0.05, 0.1, 0.15, 0.2, 0.25};
    std::vector<std::vector<double>> parameter_results;
    std::vector<double> initial_state = {1.0, 1.0};
    
    diffeq::examples::ODEStdExecution<std::vector<double>, double>::parameter_sweep(
        parameterized_system, initial_state, decay_rates, parameter_results, dt, steps);
    
    std::cout << "Parameter sweep results:" << std::endl;
    for (size_t i = 0; i < decay_rates.size(); ++i) {
        std::cout << "  Decay rate " << decay_rates[i] << ": [" 
                  << parameter_results[i][0] << ", " << parameter_results[i][1] << "]" << std::endl;
    }
    
    std::cout << "\n=== Testing Unified Parallel Interface ===" << std::endl;
    
    // Reset initial conditions
    for (int i = 0; i < num_conditions; ++i) {
        initial_conditions[i] = {static_cast<double>(i), static_cast<double>(i) * 0.5};
    }
    
    diffeq::examples::ODEParallel<std::vector<double>, double> parallel;
    parallel.integrate_parallel(system, initial_conditions, dt, steps);
    
    std::cout << "Unified interface completed successfully!" << std::endl;
    std::cout << "Result for condition 10: [" << initial_conditions[10][0] << ", " << initial_conditions[10][1] << "]" << std::endl;
    
    std::cout << "\n=== All standard parallelism examples completed! ===" << std::endl;
    
    return 0;
}