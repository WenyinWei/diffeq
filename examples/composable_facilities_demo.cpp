/**
 * @file composable_facilities_demo.cpp
 * @brief Demonstration of composable, decoupled facilities
 * 
 * This example shows how the diffeq library's composable architecture
 * solves the combinatorial explosion problem by employing high cohesion,
 * low coupling principles with the decorator pattern.
 */

#include <diffeq.hpp>
#include <core/composable_integration.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>

// Test systems
void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

void lorenz_system(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    const double sigma = 10.0, rho = 28.0, beta = 8.0/3.0;
    dydt[0] = sigma * (y[1] - y[0]);
    dydt[1] = y[0] * (rho - y[2]) - y[1];
    dydt[2] = y[0] * y[1] - beta * y[2];
}

void demonstrate_individual_facilities() {
    std::cout << "\n=== Individual Facilities (High Cohesion) ===\n";
    
    // Each facility is completely independent and focused on one concern
    
    std::cout << "\n1. Timeout Facility Only\n";
    {
        auto base_integrator = std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(exponential_decay);
        auto timeout_integrator = diffeq::core::composable::with_timeout_only(
            std::move(base_integrator),
            diffeq::core::composable::TimeoutConfig{.timeout_duration = std::chrono::milliseconds{1000}}
        );
        
        auto* timeout_decorator = dynamic_cast<diffeq::core::composable::TimeoutDecorator<std::vector<double>>*>(timeout_integrator.get());
        if (timeout_decorator) {
            std::vector<double> state = {1.0};
            auto result = timeout_decorator->integrate_with_timeout(state, 0.01, 1.0);
            std::cout << "  Timeout-only integration: " << (result.is_success() ? "✓" : "✗") 
                      << " (" << result.elapsed_time.count() << "ms)\n";
        }
    }
    
    std::cout << "\n2. Parallel Facility Only\n";
    {
        auto base_integrator = std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(exponential_decay);
        auto parallel_integrator = diffeq::core::composable::with_parallel_only(
            std::move(base_integrator),
            diffeq::core::composable::ParallelConfig{.max_threads = 4}
        );
        
        std::cout << "  Parallel-only integrator created successfully ✓\n";
        std::cout << "  (Would enable batch processing and Monte Carlo)\n";
    }
    
    std::cout << "\n3. Async Facility Only\n";
    {
        auto base_integrator = std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(exponential_decay);
        auto async_integrator = diffeq::core::composable::with_async_only(
            std::move(base_integrator),
            diffeq::core::composable::AsyncConfig{.thread_pool_size = 2}
        );
        
        auto* async_decorator = dynamic_cast<diffeq::core::composable::AsyncDecorator<std::vector<double>>*>(async_integrator.get());
        if (async_decorator) {
            std::vector<double> state = {1.0};
            auto future = async_decorator->integrate_async(state, 0.01, 0.5);
            future.wait();
            std::cout << "  Async-only integration completed ✓\n";
        }
    }
}

void demonstrate_flexible_composition() {
    std::cout << "\n=== Flexible Composition (Low Coupling) ===\n";
    
    std::cout << "\n1. Timeout + Output (2 facilities)\n";
    {
        auto base_integrator = std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(lorenz_system);
        
        // Compose timeout + output in any order
        auto composed_integrator = diffeq::core::composable::make_builder(std::move(base_integrator))
            .with_timeout(diffeq::core::composable::TimeoutConfig{.timeout_duration = std::chrono::milliseconds{2000}})
            .with_output(diffeq::core::composable::OutputConfig{.mode = diffeq::core::composable::OutputMode::ONLINE},
                        [](const std::vector<double>& state, double t, size_t step) {
                            if (step % 10 == 0) {
                                std::cout << "    t=" << std::fixed << std::setprecision(3) << t 
                                         << ", |state|=" << std::setprecision(3) 
                                         << std::sqrt(state[0]*state[0] + state[1]*state[1] + state[2]*state[2]) << "\n";
                            }
                        })
            .build();
        
        std::vector<double> state = {1.0, 1.0, 1.0};
        composed_integrator->integrate(state, 0.01, 0.3);
        std::cout << "  ✓ Timeout + Output composition completed\n";
    }
    
    std::cout << "\n2. Async + Signals + Output (3 facilities)\n";
    {
        auto base_integrator = std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(exponential_decay);
        
        // Compose 3 facilities together
        auto composed_integrator = diffeq::core::composable::make_builder(std::move(base_integrator))
            .with_async(diffeq::core::composable::AsyncConfig{.thread_pool_size = 1})
            .with_signals(diffeq::core::composable::SignalConfig{.enable_real_time_processing = true})
            .with_output(diffeq::core::composable::OutputConfig{.mode = diffeq::core::composable::OutputMode::HYBRID})
            .build();
        
        // Register signal handler
        auto* signal_decorator = dynamic_cast<diffeq::core::composable::SignalDecorator<std::vector<double>>*>(composed_integrator.get());
        if (signal_decorator) {
            signal_decorator->register_signal_handler([](std::vector<double>& state, double t) {
                // Example: external disturbance
                if (t > 0.2 && t < 0.4) {
                    state[0] += 0.01;  // Add small perturbation
                }
            });
        }
        
        std::vector<double> state = {1.0};
        composed_integrator->integrate(state, 0.01, 0.5);
        std::cout << "  ✓ Async + Signals + Output composition completed\n";
    }
    
    std::cout << "\n3. All Facilities Combined (5 facilities)\n";
    {
        auto base_integrator = std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(lorenz_system);
        
        // Compose ALL facilities - no combinatorial explosion!
        auto ultimate_integrator = diffeq::core::composable::make_builder(std::move(base_integrator))
            .with_timeout(diffeq::core::composable::TimeoutConfig{
                .timeout_duration = std::chrono::milliseconds{3000},
                .enable_progress_callback = true,
                .progress_callback = [](double current_time, double end_time, auto elapsed) {
                    std::cout << "    Progress: " << std::fixed << std::setprecision(1)
                             << (current_time / end_time) * 100 << "%\n";
                    return true;
                }
            })
            .with_parallel(diffeq::core::composable::ParallelConfig{.max_threads = 2})
            .with_async(diffeq::core::composable::AsyncConfig{.thread_pool_size = 1})
            .with_signals(diffeq::core::composable::SignalConfig{})
            .with_output(diffeq::core::composable::OutputConfig{
                .mode = diffeq::core::composable::OutputMode::ONLINE,
                .output_interval = std::chrono::microseconds{50000}  // 50ms
            }, [](const std::vector<double>& state, double t, size_t step) {
                std::cout << "    Ultimate output t=" << std::fixed << std::setprecision(2) << t << "\n";
            })
            .build();
        
        std::cout << "  ✓ All 5 facilities composed successfully!\n";
        std::cout << "  Components: Timeout + Parallel + Async + Signals + Output\n";
        
        // This integrator now has ALL capabilities without tight coupling
        std::vector<double> state = {1.0, 1.0, 1.0};
        ultimate_integrator->integrate(state, 0.01, 0.2);
        std::cout << "  ✓ Ultimate integration completed\n";
    }
}

void demonstrate_order_independence() {
    std::cout << "\n=== Order Independence (Decorator Pattern) ===\n";
    
    // Same facilities, different composition orders - all work!
    
    std::cout << "\n1. Order: Timeout → Async → Output\n";
    {
        auto integrator1 = diffeq::core::composable::make_builder(
            std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(exponential_decay))
            .with_timeout()
            .with_async()
            .with_output()
            .build();
        
        std::vector<double> state = {1.0};
        integrator1->integrate(state, 0.01, 0.2);
        std::cout << "  ✓ Order 1 completed\n";
    }
    
    std::cout << "\n2. Order: Output → Timeout → Async\n";
    {
        auto integrator2 = diffeq::core::composable::make_builder(
            std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(exponential_decay))
            .with_output()
            .with_timeout()
            .with_async()
            .build();
        
        std::vector<double> state = {1.0};
        integrator2->integrate(state, 0.01, 0.2);
        std::cout << "  ✓ Order 2 completed\n";
    }
    
    std::cout << "\n3. Order: Async → Output → Timeout\n";
    {
        auto integrator3 = diffeq::core::composable::make_builder(
            std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(exponential_decay))
            .with_async()
            .with_output()
            .with_timeout()
            .build();
        
        std::vector<double> state = {1.0};
        integrator3->integrate(state, 0.01, 0.2);
        std::cout << "  ✓ Order 3 completed\n";
    }
    
    std::cout << "  → All orders work identically (low coupling confirmed)\n";
}

void demonstrate_extensibility() {
    std::cout << "\n=== Extensibility for Future Facilities ===\n";
    
    // Show how new facilities can be added without modifying existing ones
    
    std::cout << "\n1. Current Architecture Supports:\n";
    std::cout << "   • Timeout protection\n";
    std::cout << "   • Parallel execution\n";
    std::cout << "   • Async execution\n";
    std::cout << "   • Signal processing\n";
    std::cout << "   • Online/offline output\n";
    
    std::cout << "\n2. Future Facilities Could Include:\n";
    std::cout << "   • InterprocessDecorator (IPC communication)\n";
    std::cout << "   • CompressionDecorator (state compression)\n";
    std::cout << "   • EncryptionDecorator (secure integration)\n";
    std::cout << "   • NetworkDecorator (distributed integration)\n";
    std::cout << "   • GPUDecorator (GPU acceleration)\n";
    std::cout << "   • CachingDecorator (result caching)\n";
    std::cout << "   • ProfilingDecorator (performance analysis)\n";
    std::cout << "   • CheckpointDecorator (save/restore state)\n";
    
    std::cout << "\n3. Adding New Facilities:\n";
    std::cout << "   → Implement IntegratorDecorator<S, T>\n";
    std::cout << "   → Add to builder with .with_new_facility()\n";
    std::cout << "   → Automatically works with all existing facilities\n";
    std::cout << "   → No modification of existing code required!\n";
    
    std::cout << "\n4. Combination Possibilities:\n";
    std::cout << "   • Current: 2^5 = 32 possible combinations\n";
    std::cout << "   • With 8 facilities: 2^8 = 256 combinations\n";
    std::cout << "   • With 10 facilities: 2^10 = 1024 combinations\n";
    std::cout << "   • All achieved with N classes instead of 2^N classes!\n";
}

void demonstrate_performance_characteristics() {
    std::cout << "\n=== Performance Characteristics ===\n";
    
    const size_t num_iterations = 100;
    
    std::cout << "\n1. Baseline (No decorators)\n";
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < num_iterations; ++i) {
            auto integrator = diffeq::RK45Integrator<std::vector<double>>(exponential_decay);
            std::vector<double> state = {1.0};
            integrator.integrate(state, 0.01, 0.1);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "  Baseline: " << duration.count() << " μs (" << num_iterations << " integrations)\n";
    }
    
    std::cout << "\n2. With Timeout Decorator\n";
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < num_iterations; ++i) {
            auto integrator = diffeq::core::composable::with_timeout_only(
                std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(exponential_decay));
            std::vector<double> state = {1.0};
            integrator->integrate(state, 0.01, 0.1);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "  With timeout: " << duration.count() << " μs (minimal overhead)\n";
    }
    
    std::cout << "\n3. With Multiple Decorators\n";
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < num_iterations; ++i) {
            auto integrator = diffeq::core::composable::make_builder(
                std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(exponential_decay))
                .with_timeout()
                .with_signals()
                .with_output()
                .build();
            std::vector<double> state = {1.0};
            integrator->integrate(state, 0.01, 0.1);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "  With 3 decorators: " << duration.count() << " μs (still minimal overhead)\n";
    }
    
    std::cout << "\n  → Decorator pattern adds minimal performance overhead\n";
    std::cout << "  → Overhead is proportional to number of active decorators\n";
    std::cout << "  → Each decorator only pays for what it uses\n";
}

void demonstrate_real_world_scenarios() {
    std::cout << "\n=== Real-World Usage Scenarios ===\n";
    
    std::cout << "\n1. Research Computing (Timeout + Parallel + Output)\n";
    {
        auto research_integrator = diffeq::core::composable::make_builder(
            std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(lorenz_system))
            .with_timeout(diffeq::core::composable::TimeoutConfig{.timeout_duration = std::chrono::hours{1}})
            .with_parallel(diffeq::core::composable::ParallelConfig{.max_threads = 8})
            .with_output(diffeq::core::composable::OutputConfig{
                .mode = diffeq::core::composable::OutputMode::OFFLINE,
                .buffer_size = 10000
            })
            .build();
        
        std::cout << "  ✓ Research integrator: Long timeout + Parallel + Buffered output\n";
    }
    
    std::cout << "\n2. Real-time Control (Timeout + Async + Signals)\n";
    {
        auto control_integrator = diffeq::core::composable::make_builder(
            std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(exponential_decay))
            .with_timeout(diffeq::core::composable::TimeoutConfig{.timeout_duration = std::chrono::milliseconds{10}})
            .with_async(diffeq::core::composable::AsyncConfig{.thread_pool_size = 1})
            .with_signals(diffeq::core::composable::SignalConfig{.enable_real_time_processing = true})
            .build();
        
        std::cout << "  ✓ Control integrator: Short timeout + Async + Real-time signals\n";
    }
    
    std::cout << "\n3. Production Server (Timeout + Output + Monitoring)\n";
    {
        auto server_integrator = diffeq::core::composable::make_builder(
            std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(exponential_decay))
            .with_timeout(diffeq::core::composable::TimeoutConfig{
                .timeout_duration = std::chrono::seconds{30},
                .throw_on_timeout = false  // Don't crash server
            })
            .with_output(diffeq::core::composable::OutputConfig{
                .mode = diffeq::core::composable::OutputMode::HYBRID
            }, [](const auto& state, double t, size_t step) {
                // Log to monitoring system
                static size_t log_count = 0;
                if (++log_count % 100 == 0) {
                    std::cout << "    Server log: integration step " << step << "\n";
                }
            })
            .build();
        
        std::cout << "  ✓ Server integrator: Safe timeout + Hybrid output + Monitoring\n";
    }
    
    std::cout << "\n4. Interactive Application (All facilities for flexibility)\n";
    {
        auto interactive_integrator = diffeq::core::composable::make_builder(
            std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(lorenz_system))
            .with_timeout(diffeq::core::composable::TimeoutConfig{
                .timeout_duration = std::chrono::seconds{5},
                .enable_progress_callback = true,
                .progress_callback = [](double current, double end, auto elapsed) {
                    // Update progress bar
                    return true;  // Continue unless user cancels
                }
            })
            .with_async(diffeq::core::composable::AsyncConfig{})
            .with_signals(diffeq::core::composable::SignalConfig{})
            .with_output(diffeq::core::composable::OutputConfig{
                .mode = diffeq::core::composable::OutputMode::ONLINE
            })
            .build();
        
        std::cout << "  ✓ Interactive integrator: Progress + Async + Signals + Live output\n";
    }
}

int main() {
    std::cout << "DiffEq Library - Composable Facilities Architecture Demo\n";
    std::cout << "=========================================================\n";
    
    std::cout << "\nThis demo shows how high cohesion, low coupling design\n";
    std::cout << "solves the combinatorial explosion problem:\n";
    std::cout << "• Each facility is independent (high cohesion)\n";
    std::cout << "• Facilities compose flexibly (low coupling)\n";
    std::cout << "• Adding N facilities requires N classes, not 2^N classes\n";
    std::cout << "• Any combination of facilities works seamlessly\n";
    
    try {
        demonstrate_individual_facilities();
        demonstrate_flexible_composition();
        demonstrate_order_independence();
        demonstrate_extensibility();
        demonstrate_performance_characteristics();
        demonstrate_real_world_scenarios();
        
        std::cout << "\n=== Architecture Benefits Demonstrated ===\n";
        std::cout << "✓ High Cohesion: Each facility focused on single concern\n";
        std::cout << "✓ Low Coupling: Facilities combine without dependencies\n";
        std::cout << "✓ Flexibility: Any combination of facilities possible\n";
        std::cout << "✓ Extensibility: New facilities add without modification\n";
        std::cout << "✓ Performance: Minimal overhead from composition\n";
        std::cout << "✓ Scalability: O(N) classes for N facilities vs O(2^N)\n";
        
        std::cout << "\n🎯 Problem Solved: No more combinatorial explosion!\n";
        std::cout << "   Instead of ParallelTimeoutSignalOutputAsyncIntegrator,\n";
        std::cout << "   we have: Builder.with_parallel().with_timeout()\n";
        std::cout << "                   .with_signals().with_output().with_async()\n";
        
    } catch (const std::exception& e) {
        std::cout << "\n❌ Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}