# Examples and Tutorials

This section contains practical examples and tutorials for using the DiffEq library.

## 🎯 Getting Started Examples

### Basic ODE Integration

```cpp
#include <diffeq/integrators/ode/rk4.hpp>
#include <iostream>
#include <vector>

int main() {
    // Define a simple exponential decay system: dy/dt = -0.1 * y
    auto system = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        dydt[0] = -0.1 * y[0];
    };
    
    // Initial conditions
    std::vector<double> y = {1.0};
    double t = 0.0;
    double dt = 0.01;
    
    // Create RK4 integrator
    diffeq::integrators::rk4<std::vector<double>> integrator;
    
    // Integrate for 100 steps
    for (int i = 0; i < 100; ++i) {
        integrator.step(system, y, t, dt);
        std::cout << "t = " << t << ", y = " << y[0] << std::endl;
    }
    
    return 0;
}
```

### Adaptive Step Size Control

```cpp
#include <diffeq/integrators/ode/rk45.hpp>
#include <iostream>

int main() {
    // Stiff van der Pol oscillator
    auto vanderpol = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        double mu = 10.0;  // Stiffness parameter
        dydt[0] = y[1];
        dydt[1] = mu * (1.0 - y[0]*y[0]) * y[1] - y[0];
    };
    
    // Initial conditions
    std::vector<double> y = {2.0, 0.0};
    double t = 0.0;
    double t_end = 20.0;
    double dt = 0.1;
    
    // Create adaptive integrator
    diffeq::integrators::rk45<std::vector<double>> integrator;
    integrator.set_tolerances(1e-8, 1e-6);  // abs_tol, rel_tol
    
    // Integrate to end time
    integrator.integrate(vanderpol, y, t, t_end, dt);
    
    std::cout << "Final state: y[0] = " << y[0] << ", y[1] = " << y[1] << std::endl;
    
    return 0;
}
```

## 🎲 Stochastic Differential Equations

### Euler-Maruyama Integration

```cpp
#include <diffeq/integrators/sde/euler_maruyama.hpp>
#include <random>
#include <iostream>

int main() {
    // Geometric Brownian motion: dX = μX dt + σX dW
    auto drift = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        double mu = 0.05;  // Drift coefficient
        dydt[0] = mu * y[0];
    };
    
    auto diffusion = [](double t, const std::vector<double>& y, std::vector<double>& dgdt) {
        double sigma = 0.2;  // Volatility
        dgdt[0] = sigma * y[0];
    };
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal(0.0, 1.0);
    
    // Initial conditions
    std::vector<double> y = {100.0};  // Initial stock price
    double t = 0.0;
    double dt = 0.01;
    
    // Create SDE integrator
    diffeq::integrators::euler_maruyama<std::vector<double>> integrator;
    
    // Integrate for 1000 steps
    for (int i = 0; i < 1000; ++i) {
        double dW = normal(gen) * std::sqrt(dt);  // Brownian increment
        integrator.step(drift, diffusion, y, t, dt, dW);
        
        if (i % 100 == 0) {
            std::cout << "t = " << t << ", X = " << y[0] << std::endl;
        }
    }
    
    return 0;
}
```

## 🚀 Performance Examples

### Parallel Integration

```cpp
#include <diffeq/integrators/ode/rk4.hpp>
#include <execution>
#include <vector>
#include <iostream>
#include <chrono>

int main() {
    // System of equations
    auto system = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        dydt[0] = -0.1 * y[0] + 0.2 * y[1];
        dydt[1] = -0.2 * y[1] + 0.1 * y[0];
    };
    
    // Create many initial conditions
    const int num_conditions = 10000;
    std::vector<std::vector<double>> initial_conditions(num_conditions);
    for (int i = 0; i < num_conditions; ++i) {
        initial_conditions[i] = {static_cast<double>(i), static_cast<double>(i+1)};
    }
    
    // Create integrator
    diffeq::integrators::rk4<std::vector<double>> integrator;
    
    // Parallel integration
    auto start = std::chrono::high_resolution_clock::now();
    
    std::for_each(std::execution::par, 
                  initial_conditions.begin(), 
                  initial_conditions.end(),
                  [&](std::vector<double>& state) {
                      double t = 0.0;
                      double dt = 0.01;
                      for (int step = 0; step < 1000; ++step) {
                          integrator.step(system, state, t, dt);
                      }
                  });
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Parallel integration of " << num_conditions 
              << " systems took " << duration.count() << " ms" << std::endl;
    
    return 0;
}
```

## 🎯 Advanced Examples

### Custom State Types

```cpp
#include <diffeq/integrators/ode/rk4.hpp>
#include <iostream>

// Custom 3D vector state
struct Vector3D {
    double x, y, z;
    
    Vector3D() : x(0), y(0), z(0) {}
    Vector3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    
    Vector3D operator+(const Vector3D& other) const {
        return Vector3D(x + other.x, y + other.y, z + other.z);
    }
    
    Vector3D operator*(double scalar) const {
        return Vector3D(x * scalar, y * scalar, z * scalar);
    }
    
    Vector3D& operator+=(const Vector3D& other) {
        x += other.x; y += other.y; z += other.z;
        return *this;
    }
    
    Vector3D& operator*=(double scalar) {
        x *= scalar; y *= scalar; z *= scalar;
        return *this;
    }
};

int main() {
    // Lorenz system
    auto lorenz = [](double t, const Vector3D& state, Vector3D& derivative) {
        double sigma = 10.0;
        double rho = 28.0;
        double beta = 8.0/3.0;
        
        derivative.x = sigma * (state.y - state.x);
        derivative.y = state.x * (rho - state.z) - state.y;
        derivative.z = state.x * state.y - beta * state.z;
    };
    
    // Initial conditions
    Vector3D state(1.0, 1.0, 1.0);
    double t = 0.0;
    double dt = 0.01;
    
    // Create integrator
    diffeq::integrators::rk4<Vector3D> integrator;
    
    // Integrate Lorenz system
    for (int i = 0; i < 10000; ++i) {
        integrator.step(lorenz, state, t, dt);
        
        if (i % 1000 == 0) {
            std::cout << "t = " << t << ", x = " << state.x 
                      << ", y = " << state.y << ", z = " << state.z << std::endl;
        }
    }
    
    return 0;
}
```

### Event Detection

```cpp
#include <diffeq/integrators/ode/rk45.hpp>
#include <iostream>
#include <cmath>

int main() {
    // Bouncing ball with event detection
    auto ball = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        double g = 9.81;  // Gravity
        dydt[0] = y[1];   // dx/dt = v
        dydt[1] = -g;     // dv/dt = -g
    };
    
    // Initial conditions: position = 10m, velocity = 0
    std::vector<double> y = {10.0, 0.0};
    double t = 0.0;
    double dt = 0.01;
    
    // Create integrator
    diffeq::integrators::rk45<std::vector<double>> integrator;
    
    // Integrate until ball hits ground
    while (y[0] > 0.0) {
        integrator.step(ball, y, t, dt);
        
        // Check for ground collision
        if (y[0] <= 0.0) {
            std::cout << "Ball hit ground at t = " << t << std::endl;
            
            // Bounce: reverse velocity with damping
            y[0] = 0.0;
            y[1] = -0.8 * y[1];  // Coefficient of restitution = 0.8
            
            std::cout << "Ball bounced with velocity = " << y[1] << std::endl;
        }
        
        // Stop if velocity is too small
        if (std::abs(y[1]) < 0.1 && y[0] < 0.1) {
            std::cout << "Ball came to rest at t = " << t << std::endl;
            break;
        }
    }
    
    return 0;
}
```

## 📊 Benchmarking Examples

### Integrator Comparison

```cpp
#include <diffeq/integrators/ode/euler.hpp>
#include <diffeq/integrators/ode/rk4.hpp>
#include <diffeq/integrators/ode/rk45.hpp>
#include <chrono>
#include <iostream>

int main() {
    // Test system: harmonic oscillator
    auto harmonic = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        dydt[0] = y[1];
        dydt[1] = -y[0];
    };
    
    std::vector<double> y_exact = {1.0, 0.0};
    double t_end = 10.0;
    int steps = 10000;
    double dt = t_end / steps;
    
    // Test different integrators
    std::vector<std::string> methods = {"Euler", "RK4", "RK45"};
    
    for (const auto& method : methods) {
        std::vector<double> y = {1.0, 0.0};
        double t = 0.0;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        if (method == "Euler") {
            diffeq::integrators::euler<std::vector<double>> integrator;
            for (int i = 0; i < steps; ++i) {
                integrator.step(harmonic, y, t, dt);
            }
        } else if (method == "RK4") {
            diffeq::integrators::rk4<std::vector<double>> integrator;
            for (int i = 0; i < steps; ++i) {
                integrator.step(harmonic, y, t, dt);
            }
        } else if (method == "RK45") {
            diffeq::integrators::rk45<std::vector<double>> integrator;
            integrator.set_tolerances(1e-8, 1e-6);
            integrator.integrate(harmonic, y, 0.0, t_end, dt);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Calculate error
        double exact_x = std::cos(t_end);
        double error = std::abs(y[0] - exact_x);
        
        std::cout << method << ": " << duration.count() << " μs, error = " << error << std::endl;
    }
    
    return 0;
}
```

## 🔗 Real-World Applications

### Financial Modeling

```cpp
#include <diffeq/integrators/sde/euler_maruyama.hpp>
#include <random>
#include <iostream>
#include <vector>

// Black-Scholes model for option pricing
int main() {
    double S0 = 100.0;    // Initial stock price
    double K = 105.0;     // Strike price
    double T = 1.0;       // Time to expiration
    double r = 0.05;      // Risk-free rate
    double sigma = 0.2;   // Volatility
    
    // SDE: dS = rS dt + σS dW
    auto drift = [r](double t, const std::vector<double>& S, std::vector<double>& dSdt) {
        dSdt[0] = r * S[0];
    };
    
    auto diffusion = [sigma](double t, const std::vector<double>& S, std::vector<double>& dSdW) {
        dSdW[0] = sigma * S[0];
    };
    
    // Monte Carlo simulation
    int num_simulations = 100000;
    double payoff_sum = 0.0;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal(0.0, 1.0);
    
    diffeq::integrators::euler_maruyama<std::vector<double>> integrator;
    
    for (int sim = 0; sim < num_simulations; ++sim) {
        std::vector<double> S = {S0};
        double t = 0.0;
        double dt = T / 1000.0;
        
        // Simulate price path
        for (int i = 0; i < 1000; ++i) {
            double dW = normal(gen) * std::sqrt(dt);
            integrator.step(drift, diffusion, S, t, dt, dW);
        }
        
        // Calculate payoff
        double payoff = std::max(S[0] - K, 0.0);
        payoff_sum += payoff;
    }
    
    // Option price
    double option_price = std::exp(-r * T) * payoff_sum / num_simulations;
    std::cout << "Call option price: " << option_price << std::endl;
    
    return 0;
}
```

## 🔗 See Also

- [API Documentation](../api/README.md) - Complete API reference
- [Performance Guide](../performance/README.md) - Optimization techniques
- [Main Documentation](../index.md) - Library overview

## 📝 Running the Examples

To compile and run these examples:

```bash
# Using xmake
xmake build examples

# Manual compilation
g++ -std=c++20 -I./include -O3 example.cpp -o example
./example
```

Make sure you have:
- C++20 compatible compiler
- Eigen library (for some examples)
- OpenMP (for parallel examples) 