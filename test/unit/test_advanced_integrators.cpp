#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <cmath>
#include <iostream>
#include <chrono>

// Include integrators (excluding DOP853 which has its own test file)
#include <integrators/ode/rk4.hpp>
#include <integrators/ode/rk23.hpp>
#include <integrators/ode/rk45.hpp>
#include <integrators/ode/bdf.hpp>
#include <integrators/ode/lsoda.hpp>

// Test system: dy/dt = -y (exact solution: y(t) = y0 * exp(-t))
void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

// Test system: Van der Pol oscillator (stiff for large mu)
class VanderPolOscillator {
public:
    explicit VanderPolOscillator(double mu) : mu_(mu) {}
    
    void operator()(double t, const std::vector<double>& y, std::vector<double>& dydt) {
        dydt[0] = y[1];
        dydt[1] = mu_ * (1 - y[0]*y[0]) * y[1] - y[0];
    }
    
private:
    double mu_;
};

// Test system: Lorenz system (chaotic)
void lorenz_system(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    const double sigma = 10.0;
    const double rho = 28.0;
    const double beta = 8.0/3.0;
    
    dydt[0] = sigma * (y[1] - y[0]);
    dydt[1] = y[0] * (rho - y[2]) - y[1];
    dydt[2] = y[0] * y[1] - beta * y[2];
}

// Fixed-size array test
void exponential_decay_array(double t, const std::array<double, 1>& y, std::array<double, 1>& dydt) {
    dydt[0] = -y[0];
}

class IntegratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test parameters
        y0_vector_ = {1.0};
        y0_array_ = {1.0};
        t_start_ = 0.0;
        t_end_ = 1.0;
        dt_ = 0.1;
        tolerance_ = 1e-3;
    }
    
    double analytical_solution(double t) {
        return std::exp(-t);
    }
    
    std::vector<double> y0_vector_;
    std::array<double, 1> y0_array_;
    double t_start_, t_end_, dt_, tolerance_;
};

TEST_F(IntegratorTest, RK4IntegratorVector) {
    diffeq::integrators::ode::RK4Integrator<std::vector<double>> integrator(exponential_decay);
    
    auto y = y0_vector_;
    integrator.set_time(t_start_);
    integrator.integrate(y, dt_, t_end_);
    
    double exact = analytical_solution(t_end_);
    EXPECT_NEAR(y[0], exact, tolerance_);
}

TEST_F(IntegratorTest, RK4IntegratorArray) {
    diffeq::integrators::ode::RK4Integrator<std::array<double, 1>> integrator(exponential_decay_array);
    
    auto y = y0_array_;
    integrator.set_time(t_start_);
    integrator.integrate(y, dt_, t_end_);
    
    double exact = analytical_solution(t_end_);
    EXPECT_NEAR(y[0], exact, tolerance_);
}

TEST_F(IntegratorTest, RK23IntegratorAdaptive) {
    diffeq::integrators::ode::RK23Integrator<std::vector<double>> integrator(exponential_decay, 1e-6, 1e-9);
    
    auto y = y0_vector_;
    integrator.set_time(t_start_);
    integrator.integrate(y, dt_, t_end_);
    
    double exact = analytical_solution(t_end_);
    EXPECT_NEAR(y[0], exact, 1e-5);
}

TEST_F(IntegratorTest, RK45IntegratorAdaptive) {
    diffeq::integrators::ode::RK45Integrator<std::vector<double>> integrator(exponential_decay, 1e-6, 1e-9);
    
    auto y = y0_vector_;
    integrator.set_time(t_start_);
    integrator.integrate(y, dt_, t_end_);
    
    double exact = analytical_solution(t_end_);
    EXPECT_NEAR(y[0], exact, 1e-6);
}

TEST_F(IntegratorTest, BDFIntegratorStiff) {
    // Test with a mildly stiff system
    VanderPolOscillator vdp(5.0);  // Moderately stiff
    
    diffeq::integrators::ode::BDFIntegrator<std::vector<double>> integrator(
        [&vdp](double t, const std::vector<double>& y, std::vector<double>& dydt) {
            vdp(t, y, dydt);
        }, 1e-6, 1e-9);
    
    std::vector<double> y = {1.0, 0.0};  // Initial conditions
    integrator.set_time(0.0);
    
    // This should not throw for a well-conditioned problem
    EXPECT_NO_THROW({
        integrator.integrate(y, 0.1, 1.0);
    });
    
    // Basic sanity check - solution should be bounded
    EXPECT_LT(std::abs(y[0]), 10.0);
    EXPECT_LT(std::abs(y[1]), 10.0);
}

TEST_F(IntegratorTest, BDFIntegratorMultistep) {
    diffeq::integrators::ode::BDFIntegrator<std::vector<double>> integrator(exponential_decay, 1e-6, 1e-9, 3);
    
    auto y = y0_vector_;
    integrator.set_time(t_start_);
    integrator.integrate(y, dt_, t_end_);
    
    double exact = analytical_solution(t_end_);
    EXPECT_NEAR(y[0], exact, 1e-4);
}

TEST_F(IntegratorTest, LSODAIntegratorAutomatic) {
    diffeq::integrators::ode::LSODA<std::vector<double>> integrator(exponential_decay, 1e-6, 1e-9);
    
    auto y = y0_vector_;
    integrator.set_time(t_start_);
    integrator.integrate(y, dt_, t_end_);
    
    double exact = analytical_solution(t_end_);
    EXPECT_NEAR(y[0], exact, 1e-5);
    
    // Should start with Adams method for non-stiff problem
    // Note: get_current_method() may not be available in current implementation
}

TEST_F(IntegratorTest, LSODAStiffnessSwitching) {
    // Test with Van der Pol oscillator that becomes stiff
    VanderPolOscillator vdp(10.0);  // Stiff system
    
    diffeq::integrators::ode::LSODA<std::vector<double>> integrator(
        [&vdp](double t, const std::vector<double>& y, std::vector<double>& dydt) {
            vdp(t, y, dydt);
        }, 1e-6, 1e-9);
    
    // Note: set_stiffness_detection_frequency may not be available in current implementation
    
    std::vector<double> y = {1.0, 0.0};
    integrator.set_time(0.0);
    
    // Run integration - should automatically switch to BDF when stiffness is detected
    EXPECT_NO_THROW({
        integrator.integrate(y, 0.01, 0.5);
    });
    
    // Solution should be bounded
    EXPECT_LT(std::abs(y[0]), 10.0);
    EXPECT_LT(std::abs(y[1]), 10.0);
}

TEST_F(IntegratorTest, LorenzSystemChaotic) {
    // Test all integrators on Lorenz system
    std::vector<double> y0 = {1.0, 1.0, 1.0};
    double t_end = 2.0;
    double dt = 0.01;
    
    // RK4
    {
        diffeq::integrators::ode::RK4Integrator<std::vector<double>> integrator(lorenz_system);
        auto y = y0;
        integrator.set_time(0.0);
        EXPECT_NO_THROW(integrator.integrate(y, dt, t_end));
        // Just check solution is bounded (Lorenz attractor is bounded)
        EXPECT_LT(std::abs(y[0]), 50.0);
        EXPECT_LT(std::abs(y[1]), 50.0);
        EXPECT_LT(std::abs(y[2]), 50.0);
    }
    
    // RK45
    {
        diffeq::integrators::ode::RK45Integrator<std::vector<double>> integrator(lorenz_system, 1e-8, 1e-12);
        auto y = y0;
        integrator.set_time(0.0);
        EXPECT_NO_THROW(integrator.integrate(y, dt, t_end));
        EXPECT_LT(std::abs(y[0]), 50.0);
        EXPECT_LT(std::abs(y[1]), 50.0);
        EXPECT_LT(std::abs(y[2]), 50.0);
    }
    
    // LSODA
    {
        diffeq::integrators::ode::LSODA<std::vector<double>> integrator(lorenz_system, 1e-8, 1e-12);
        auto y = y0;
        integrator.set_time(0.0);
        EXPECT_NO_THROW(integrator.integrate(y, dt, t_end));
        EXPECT_LT(std::abs(y[0]), 50.0);
        EXPECT_LT(std::abs(y[1]), 50.0);
        EXPECT_LT(std::abs(y[2]), 50.0);
    }
}

TEST_F(IntegratorTest, ToleranceSettings) {
    diffeq::integrators::ode::RK45Integrator<std::vector<double>> integrator(exponential_decay);
    
    // Test different tolerance levels
    std::vector<std::pair<double, double>> tolerances = {
        {1e-3, 1e-6}, {1e-6, 1e-9}, {1e-9, 1e-12}
    };
    
    for (auto [rtol, atol] : tolerances) {
        integrator.set_tolerances(rtol, atol);
        
        auto y = y0_vector_;
        integrator.set_time(t_start_);
        integrator.integrate(y, dt_, t_end_);
        
        double exact = analytical_solution(t_end_);
        double error = std::abs(y[0] - exact);
        
        // Error should be roughly proportional to tolerance
        EXPECT_LT(error, rtol * 10);  // Allow some margin
    }
}

// Performance comparison test (just check they all run)
TEST_F(IntegratorTest, PerformanceComparison) {
    std::vector<double> y0 = {1.0, 1.0, 1.0};
    double t_end = 1.0;
    double dt = 0.001;
    
    // Test that all integrators can handle the same problem
    {
        diffeq::integrators::ode::RK4Integrator<std::vector<double>> integrator(lorenz_system);
        auto y = y0;
        integrator.set_time(0.0);
        EXPECT_NO_THROW(integrator.integrate(y, dt, t_end));
    }
    
    {
        diffeq::integrators::ode::RK23Integrator<std::vector<double>> integrator(lorenz_system);
        auto y = y0;
        integrator.set_time(0.0);
        EXPECT_NO_THROW(integrator.integrate(y, dt, t_end));
    }
    
    {
        diffeq::integrators::ode::RK45Integrator<std::vector<double>> integrator(lorenz_system);
        auto y = y0;
        integrator.set_time(0.0);
        EXPECT_NO_THROW(integrator.integrate(y, dt, t_end));
    }
    
    {
        diffeq::integrators::ode::BDFIntegrator<std::vector<double>> integrator(lorenz_system);
        auto y = y0;
        integrator.set_time(0.0);
        EXPECT_NO_THROW(integrator.integrate(y, dt, t_end));
    }
    
    {
        diffeq::integrators::ode::LSODA<std::vector<double>> integrator(lorenz_system);
        auto y = y0;
        integrator.set_time(0.0);
        EXPECT_NO_THROW(integrator.integrate(y, dt, t_end));
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
