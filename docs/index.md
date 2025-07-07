# DiffEq - Modern C++ Differential Equation Library

Welcome to the DiffEq library documentation. This library provides a modern, header-only C++ solution for solving ordinary differential equations (ODEs) and stochastic differential equations (SDEs).

## 🚀 Quick Start

```cpp
#include <diffeq/integrators/ode/rk4.hpp>
#include <diffeq/state.hpp>

int main() {
    // Define your ODE system
    auto system = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        dydt[0] = -0.1 * y[0]; // Simple exponential decay
    };
    
    // Initial conditions
    std::vector<double> y0 = {1.0};
    double t0 = 0.0;
    double dt = 0.01;
    
    // Create RK4 integrator
    diffeq::integrators::rk4<std::vector<double>> integrator;
    
    // Integrate forward
    integrator.step(system, y0, t0, dt);
    
    return 0;
}
```

## 📚 Documentation Sections

- **[API Documentation](api/README.md)** - Comprehensive API reference
- **[Examples](examples/README.md)** - Usage examples and tutorials
- **[Performance](performance/README.md)** - Performance analysis and optimization guides

## 🔧 Key Features

- **Modern C++20** - Leverages latest C++ features
- **Header-only** - Easy integration into projects
- **Type-safe** - Template-based design with concepts
- **High Performance** - Optimized algorithms with parallel execution support
- **Extensible** - Easy to add new integrators and features

## 🧮 Supported Integrators

### ODE Integrators
- **Euler** - Simple first-order method
- **Improved Euler** - Second-order Heun method
- **RK4** - Classic fourth-order Runge-Kutta
- **RK23** - Adaptive Runge-Kutta with embedded error estimation
- **RK45** - Adaptive Dormand-Prince method
- **DOP853** - High-order adaptive method
- **BDF** - Backward differentiation formulas for stiff systems
- **LSODA** - Automatic stiffness detection and switching

### SDE Integrators
- **Euler-Maruyama** - Basic SDE integration
- **Milstein** - Higher-order SDE method
- **SRI1** - Stochastic Runge-Kutta method
- **SRA1/SRA2** - Simplified stochastic Runge-Kutta
- **SOSRI/SOSRA** - Second-order stochastic methods
- **Implicit Euler-Maruyama** - For stiff stochastic systems

## 🔗 Navigation

- [Build Instructions](README.md#building)
- [API Reference](api/README.md)
- [Examples](examples/README.md)
- [Performance Guides](performance/README.md)
- [Contributing Guidelines](../README.md#contributing)

## 📄 License

This library is available under multiple licenses. See [LICENSE](../LICENSE) for details. 