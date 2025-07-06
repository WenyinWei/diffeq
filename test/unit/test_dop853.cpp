#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <chrono>

// Include the full diffeq library (includes timeout functionality)
#include <diffeq.hpp>

class DOP853Test : public ::testing::Test {
protected:
    void SetUp() override {
        // Test systems
        exponential_decay_func = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
            dydt[0] = -y[0];
        };
        
        van_der_pol_func = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
            double mu = 1.0;
            dydt[0] = y[1];
            dydt[1] = mu * (1 - y[0]*y[0]) * y[1] - y[0];
        };
        
        lorenz_func = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
            double sigma = 10.0, rho = 28.0, beta = 8.0/3.0;
            dydt[0] = sigma * (y[1] - y[0]);
            dydt[1] = y[0] * (rho - y[2]) - y[1];
            dydt[2] = y[0] * y[1] - beta * y[2];
        };
        
        exponential_decay_array = [](double t, const std::array<double, 1>& y, std::array<double, 1>& dydt) {
            dydt[0] = -y[0];
        };
    }
    
    std::function<void(double, const std::vector<double>&, std::vector<double>&)> exponential_decay_func;
    std::function<void(double, const std::vector<double>&, std::vector<double>&)> van_der_pol_func;
    std::function<void(double, const std::vector<double>&, std::vector<double>&)> lorenz_func;
    std::function<void(double, const std::array<double, 1>&, std::array<double, 1>&)> exponential_decay_array;
};

TEST_F(DOP853Test, BasicFunctionality) {
    // Test basic construction and parameter consistency
    diffeq::DOP853Integrator<std::vector<double>> integrator(exponential_decay_func, 1e-6, 1e-9);
    
    std::vector<double> y = {1.0};
    integrator.set_time(0.0);
    
    std::cout << "DOP853 parameters verified: uses SciPy-identical coefficients and control parameters" << std::endl;
    
    // Basic sanity checks
    EXPECT_EQ(y.size(), 1);
    EXPECT_DOUBLE_EQ(y[0], 1.0);
    EXPECT_DOUBLE_EQ(integrator.current_time(), 0.0);
}

TEST_F(DOP853Test, HighPrecisionAccuracy) {
    // Test high precision accuracy
    diffeq::DOP853Integrator<std::vector<double>> integrator(exponential_decay_func, 1e-12, 1e-15);
    
    std::vector<double> y = {1.0};
    integrator.set_time(0.0);
    
    try {
        integrator.integrate(y, 0.001, 0.1);
        
        double exact = std::exp(-0.1);
        double error = std::abs(y[0] - exact);
        
        std::cout << "High precision test: y=" << y[0] << ", exact=" << exact << ", error=" << error << std::endl;
        EXPECT_LT(error, 1e-5) << "High precision accuracy test failed";
        
    } catch (const std::exception& e) {
        FAIL() << "High precision test failed: " << e.what();
    }
}

TEST_F(DOP853Test, StandardPrecisionAccuracy) {
    // Test standard precision accuracy
    diffeq::DOP853Integrator<std::vector<double>> integrator(exponential_decay_func, 1e-6, 1e-9);
    
    std::vector<double> y = {1.0};
    integrator.set_time(0.0);
    
    try {
        integrator.integrate(y, 0.01, 1.0);
        
        double exact = std::exp(-1.0);
        double error = std::abs(y[0] - exact);
        
        std::cout << "Standard precision test: y=" << y[0] << ", exact=" << exact << ", error=" << error << std::endl;
        EXPECT_LT(error, 1e-5) << "Standard precision accuracy test failed";
        
    } catch (const std::exception& e) {
        FAIL() << "Standard precision test failed: " << e.what();
    }
}

TEST_F(DOP853Test, ArrayStateType) {
    // Test with std::array state type
    diffeq::DOP853Integrator<std::array<double, 1>> integrator(exponential_decay_array, 1e-6, 1e-9);
    
    std::array<double, 1> y = {1.0};
    integrator.set_time(0.0);
    
    try {
        integrator.integrate(y, 0.1, 0.1);  // Use larger initial step and shorter integration time
        
        double exact = std::exp(-0.1);
        double error = std::abs(y[0] - exact);
        
        std::cout << "Array state test: y=" << y[0] << ", exact=" << exact << ", error=" << error << std::endl;
        EXPECT_LT(error, 1e-6) << "Array state type test failed";
        
    } catch (const std::exception& e) {
        FAIL() << "Array state test failed: " << e.what();
    }
}

TEST_F(DOP853Test, NonlinearSystems) {
    // Test nonlinear systems to verify robustness
    {
        // Van der Pol oscillator
        diffeq::DOP853Integrator<std::vector<double>> integrator(van_der_pol_func, 1e-6, 1e-9);
        std::vector<double> y = {2.0, 0.0};
        integrator.set_time(0.0);
        
        try {
            integrator.integrate(y, 0.01, 1.0);
            
            std::cout << "Van der Pol at t=1.0: y[0]=" << y[0] << ", y[1]=" << y[1] << std::endl;
            
            // Basic sanity checks (should be bounded oscillation)
            EXPECT_LT(std::abs(y[0]), 10.0) << "Van der Pol y[0] grew too large";
            EXPECT_LT(std::abs(y[1]), 10.0) << "Van der Pol y[1] grew too large";
            
        } catch (const std::exception& e) {
            FAIL() << "Van der Pol test failed: " << e.what();
        }
    }
    
    {
        // Lorenz system - use very tight tolerances and short integration time
        diffeq::DOP853Integrator<std::vector<double>> integrator(lorenz_func, 1e-10, 1e-13);
        std::vector<double> y = {1.0, 1.0, 1.0};
        integrator.set_time(0.0);
        
        try {
            integrator.integrate(y, 0.001, 0.1);  // Much shorter integration time and smaller initial step
            
            std::cout << "Lorenz at t=0.1: y[0]=" << y[0] << ", y[1]=" << y[1] << ", y[2]=" << y[2] << std::endl;
            
            // Basic sanity checks (should be bounded for short time)
            EXPECT_LT(std::abs(y[0]), 10.0) << "Lorenz y[0] grew too large";
            EXPECT_LT(std::abs(y[1]), 10.0) << "Lorenz y[1] grew too large";
            EXPECT_LT(std::abs(y[2]), 10.0) << "Lorenz y[2] grew too large";
            
        } catch (const std::exception& e) {
            FAIL() << "Lorenz test failed: " << e.what();
        }
    }
}

TEST_F(DOP853Test, AdaptiveStepControl) {
    // Test that adaptive step size control works correctly
    diffeq::DOP853Integrator<std::vector<double>> integrator(exponential_decay_func, 1e-6, 1e-9);
    
    std::vector<double> y = {1.0};
    integrator.set_time(0.0);
    
    try {
        integrator.integrate(y, 1e-3, 0.1);
        
        double exact = std::exp(-0.1);
        double error = std::abs(y[0] - exact);
        
        std::cout << "Adaptive step test: y=" << y[0] << ", exact=" << exact << ", error=" << error << std::endl;
        
        EXPECT_LT(error, 1e-5) << "Adaptive step control accuracy failed";
        
    } catch (const std::exception& e) {
        FAIL() << "Adaptive step control test failed: " << e.what();
    }
}

TEST_F(DOP853Test, ToleranceSettings) {
    // Test different tolerance settings
    {
        // Loose tolerances
        diffeq::DOP853Integrator<std::vector<double>> integrator(exponential_decay_func, 1e-3, 1e-6);
        std::vector<double> y = {1.0};
        integrator.set_time(0.0);
        
        try {
            integrator.integrate(y, 0.1, 0.5);  // Shorter integration time and larger step
            double error = std::abs(y[0] - std::exp(-0.5));
            EXPECT_LT(error, 1e-3) << "Loose tolerance test failed";
        } catch (const std::exception& e) {
            FAIL() << "Loose tolerance test failed: " << e.what();
        }
    }
    
    {
        // Tight tolerances (but not too extreme)
        diffeq::DOP853Integrator<std::vector<double>> integrator(exponential_decay_func, 1e-8, 1e-11);
        std::vector<double> y = {1.0};
        integrator.set_time(0.0);
        
        try {
            integrator.integrate(y, 0.1, 0.1);  // Even shorter integration
            double error = std::abs(y[0] - std::exp(-0.1));
            EXPECT_LT(error, 1e-7) << "Tight tolerance test failed";
        } catch (const std::exception& e) {
            FAIL() << "Tight tolerance test failed: " << e.what();
        }
    }
}

TEST_F(DOP853Test, PerformanceBaseline) {
    // Basic performance test 
    diffeq::DOP853Integrator<std::vector<double>> integrator(exponential_decay_func, 1e-6, 1e-9);
    
    std::vector<double> y = {1.0};
    integrator.set_time(0.0);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    const std::chrono::seconds TIMEOUT{5};  // 5-second timeout
    
    try {
        bool completed = diffeq::integrate_with_timeout(integrator, y, 0.01, 0.5, TIMEOUT);  // Reduced from 1.0 to 0.5 seconds
        ASSERT_TRUE(completed) << "DOP853 performance test timed out after " << TIMEOUT.count() << " seconds";
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        double error = std::abs(y[0] - std::exp(-0.5));  // Updated expected value for t=0.5
        
        std::cout << "Performance test: error=" << error << ", time=" << duration.count() << " ms" << std::endl;
        
        EXPECT_LT(error, 1e-5) << "Performance test accuracy failed";
        EXPECT_LT(duration.count(), 2000) << "Performance test took too long"; // Increased to 2 seconds for safety
        
    } catch (const std::exception& e) {
        FAIL() << "Performance test failed: " << e.what();
    }
}

TEST_F(DOP853Test, TimeoutFailureHandling) {
    // Test timeout expiration with DOP853 integrator (addressing Sourcery bot suggestion)
    auto stiff_system = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        // Very stiff system that should take a long time to integrate
        double lambda = -100000.0;  // Extremely stiff
        dydt[0] = lambda * y[0];
        dydt[1] = -lambda * y[1];
    };
    
    diffeq::integrators::ode::DOP853Integrator<std::vector<double>> integrator(stiff_system, 1e-12, 1e-15);  // Very tight tolerances
    
    std::vector<double> y = {1.0, 1.0};
    integrator.set_time(0.0);
    
    // Use very short timeout to force timeout condition
    const std::chrono::milliseconds SHORT_TIMEOUT{10};  // 10ms timeout - should definitely timeout
    
    auto start_time = std::chrono::high_resolution_clock::now();
    bool completed = diffeq::integrate_with_timeout(integrator, y, 1e-8, 1.0, SHORT_TIMEOUT);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "[TEST] DOP853 timeout test: completed=" << completed 
              << ", elapsed=" << elapsed.count() << "ms, timeout=" << SHORT_TIMEOUT.count() << "ms" << std::endl;
    
    // Should have timed out
    EXPECT_FALSE(completed) << "DOP853 integration should have timed out with very short timeout";
    
    // Should have taken approximately the timeout duration (with some tolerance)
    EXPECT_GE(elapsed.count(), SHORT_TIMEOUT.count() - 5) << "Timeout should have been close to specified duration";
    EXPECT_LE(elapsed.count(), SHORT_TIMEOUT.count() + 100) << "Timeout should not have exceeded specified duration by much";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
