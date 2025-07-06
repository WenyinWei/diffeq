/**
 * @file timeout_integration_demo.cpp
 * @brief Demonstration of timeout-protected integration in the diffeq library
 * 
 * This example shows how to use the timeout functionality to prevent integration
 * from hanging and to monitor progress during long-running integrations.
 */

#include <diffeq.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <functional>

// Define some test systems
void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

void lorenz_system(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    const double sigma = 10.0, rho = 28.0, beta = 8.0/3.0;
    dydt[0] = sigma * (y[1] - y[0]);
    dydt[1] = y[0] * (rho - y[2]) - y[1];
    dydt[2] = y[0] * y[1] - beta * y[2];
}

void stiff_van_der_pol(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    double mu = 100.0;  // Very stiff system
    dydt[0] = y[1];
    dydt[1] = mu * (1 - y[0]*y[0]) * y[1] - y[0];
}

void demonstrate_basic_timeout() {
    std::cout << "\n=== Basic Timeout Functionality ===\n";
    
    // Simple integration with timeout protection
    std::vector<double> y = {1.0};
    auto integrator = diffeq::RK45Integrator<std::vector<double>>(exponential_decay);
    
    // Use simple timeout function
    bool completed = diffeq::integrate_with_timeout(
        integrator, y, 0.01, 1.0, 
        std::chrono::milliseconds{1000}  // 1 second timeout
    );
    
    if (completed) {
        std::cout << "✓ Integration completed successfully\n";
        std::cout << "  Final value: " << y[0] << " (expected: " << std::exp(-1.0) << ")\n";
    } else {
        std::cout << "✗ Integration timed out\n";
    }
}

void demonstrate_timeout_wrapper() {
    std::cout << "\n=== TimeoutIntegrator Wrapper ===\n";
    
    // Create integrator
    auto integrator = diffeq::RK45Integrator<std::vector<double>>(lorenz_system);
    
    // Wrap with timeout functionality
    auto timeout_config = diffeq::TimeoutConfig{
        .timeout_duration = std::chrono::milliseconds{2000},  // 2 second timeout
        .throw_on_timeout = false,  // Don't throw, return result instead
        .enable_progress_callback = false
    };
    
    auto timeout_integrator = diffeq::make_timeout_integrator(
        std::move(integrator), timeout_config
    );
    
    // Test with Lorenz system
    std::vector<double> y = {1.0, 1.0, 1.0};
    auto result = timeout_integrator.integrate_with_timeout(y, 0.01, 1.0);
    
    std::cout << "Integration result:\n";
    std::cout << "  Completed: " << (result.completed ? "Yes" : "No") << "\n";
    std::cout << "  Elapsed time: " << result.elapsed_time.count() << " ms\n";
    std::cout << "  Final time: " << result.final_time << "\n";
    
    if (result.is_success()) {
        std::cout << "  ✓ Success! Final state: [" << y[0] << ", " << y[1] << ", " << y[2] << "]\n";
    } else if (result.is_timeout()) {
        std::cout << "  ⏰ Timed out: " << result.error_message << "\n";
    } else {
        std::cout << "  ✗ Error: " << result.error_message << "\n";
    }
}

void demonstrate_progress_monitoring() {
    std::cout << "\n=== Progress Monitoring ===\n";
    
    // Create integrator for potentially slow system
    auto integrator = diffeq::RK45Integrator<std::vector<double>>(lorenz_system);
    
    // Configure with progress monitoring
    auto timeout_config = diffeq::TimeoutConfig{
        .timeout_duration = std::chrono::milliseconds{5000},  // 5 second timeout
        .throw_on_timeout = false,
        .enable_progress_callback = true,
        .progress_interval = std::chrono::milliseconds{100},  // Check every 100ms
        .progress_callback = [](double current_time, double end_time, std::chrono::milliseconds elapsed) {
            double progress = (current_time / end_time) * 100.0;
            std::cout << "  Progress: " << std::fixed << std::setprecision(1) 
                     << progress << "% (t=" << current_time << "/" << end_time 
                     << ", elapsed=" << elapsed.count() << "ms)\n";
            
            // Continue integration (return true to continue, false to cancel)
            return true;
        }
    };
    
    auto timeout_integrator = diffeq::make_timeout_integrator(
        std::move(integrator), timeout_config
    );
    
    std::vector<double> y = {1.0, 1.0, 1.0};
    std::cout << "Starting integration with progress monitoring...\n";
    
    auto result = timeout_integrator.integrate_with_timeout(y, 0.01, 2.0);
    
    std::cout << "\nProgress monitoring result:\n";
    std::cout << "  Completed: " << (result.completed ? "Yes" : "No") << "\n";
    std::cout << "  Total elapsed time: " << result.elapsed_time.count() << " ms\n";
}

void demonstrate_exception_handling() {
    std::cout << "\n=== Exception-based Timeout Handling ===\n";
    
    // Create integrator for potentially problematic system
    auto integrator = diffeq::BDFIntegrator<std::vector<double>>(stiff_van_der_pol);
    
    // Configure to throw on timeout
    auto timeout_config = diffeq::TimeoutConfig{
        .timeout_duration = std::chrono::milliseconds{500},  // Short timeout
        .throw_on_timeout = true  // Throw exception on timeout
    };
    
    auto timeout_integrator = diffeq::make_timeout_integrator(
        std::move(integrator), timeout_config
    );
    
    std::vector<double> y = {1.0, 0.0};
    
    try {
        std::cout << "Attempting integration of stiff Van der Pol system...\n";
        auto result = timeout_integrator.integrate_with_timeout(y, 0.001, 1.0);
        
        if (result.is_success()) {
            std::cout << "  ✓ Integration completed successfully\n";
            std::cout << "  Final state: [" << y[0] << ", " << y[1] << "]\n";
        } else {
            std::cout << "  Integration did not complete: " << result.error_message << "\n";
        }
        
    } catch (const diffeq::IntegrationTimeoutException& e) {
        std::cout << "  ⏰ Caught timeout exception: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cout << "  ✗ Caught other exception: " << e.what() << "\n";
    }
}

void demonstrate_comparison() {
    std::cout << "\n=== Performance Comparison: With vs Without Timeout ===\n";
    
    // Test system that might be slow
    std::vector<double> y1 = {1.0, 1.0, 1.0};
    std::vector<double> y2 = {1.0, 1.0, 1.0};
    
    // Regular integration (no timeout protection)
    {
        auto integrator = diffeq::RK45Integrator<std::vector<double>>(lorenz_system);
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            integrator.integrate(y1, 0.01, 1.0);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "Regular integration: " << duration.count() << " ms\n";
        } catch (const std::exception& e) {
            std::cout << "Regular integration failed: " << e.what() << "\n";
        }
    }
    
    // Timeout-protected integration
    {
        auto integrator = diffeq::RK45Integrator<std::vector<double>>(lorenz_system);
        auto timeout_config = diffeq::TimeoutConfig{
            .timeout_duration = std::chrono::milliseconds{3000},
            .throw_on_timeout = false
        };
        auto timeout_integrator = diffeq::make_timeout_integrator(std::move(integrator), timeout_config);
        
        auto result = timeout_integrator.integrate_with_timeout(y2, 0.01, 1.0);
        
        std::cout << "Timeout integration: " << result.elapsed_time.count() << " ms";
        if (result.is_success()) {
            std::cout << " (✓ completed)\n";
        } else {
            std::cout << " (✗ " << result.error_message << ")\n";
        }
    }
    
    // Compare results
    if (std::abs(y1[0] - y2[0]) < 1e-10 && std::abs(y1[1] - y2[1]) < 1e-10 && std::abs(y1[2] - y2[2]) < 1e-10) {
        std::cout << "✓ Both methods produced identical results\n";
    } else {
        std::cout << "⚠ Results differ (expected for timeout-protected integration)\n";
    }
}

void demonstrate_different_integrators() {
    std::cout << "\n=== Timeout with Different Integrator Types ===\n";
    
    std::vector<double> y = {1.0};
    double dt = 0.01, t_end = 1.0;
    auto timeout = std::chrono::milliseconds{1000};
    
    // Test with different integrator types
    std::vector<std::pair<std::string, std::function<bool()>>> integrator_tests = {
        {"RK4", [&]() {
            auto integrator = diffeq::RK4Integrator<std::vector<double>>(exponential_decay);
            auto y_copy = y;
            return diffeq::integrate_with_timeout(integrator, y_copy, dt, t_end, timeout);
        }},
        {"RK23", [&]() {
            auto integrator = diffeq::RK23Integrator<std::vector<double>>(exponential_decay);
            auto y_copy = y;
            return diffeq::integrate_with_timeout(integrator, y_copy, dt, t_end, timeout);
        }},
        {"RK45", [&]() {
            auto integrator = diffeq::RK45Integrator<std::vector<double>>(exponential_decay);
            auto y_copy = y;
            return diffeq::integrate_with_timeout(integrator, y_copy, dt, t_end, timeout);
        }},
        {"DOP853", [&]() {
            auto integrator = diffeq::DOP853Integrator<std::vector<double>>(exponential_decay);
            auto y_copy = y;
            return diffeq::integrate_with_timeout(integrator, y_copy, dt, t_end, timeout);
        }}
    };
    
    for (const auto& [name, test] : integrator_tests) {
        std::cout << "  " << name << ": ";
        if (test()) {
            std::cout << "✓ Completed\n";
        } else {
            std::cout << "✗ Timed out\n";
        }
    }
}

int main() {
    std::cout << "DiffEq Library - Timeout Integration Demo\n";
    std::cout << "==========================================\n";
    
    demonstrate_basic_timeout();
    demonstrate_timeout_wrapper();
    demonstrate_progress_monitoring();
    demonstrate_exception_handling();
    demonstrate_comparison();
    demonstrate_different_integrators();
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "The diffeq library provides comprehensive timeout functionality:\n";
    std::cout << "1. Simple timeout function: diffeq::integrate_with_timeout()\n";
    std::cout << "2. Full-featured wrapper: diffeq::TimeoutIntegrator\n";
    std::cout << "3. Progress monitoring with callbacks\n";
    std::cout << "4. Exception-based error handling\n";
    std::cout << "5. Compatible with all integrator types\n";
    std::cout << "\nThis prevents hanging in production applications and enables\n";
    std::cout << "robust real-time systems with predictable behavior.\n";
    
    return 0;
}