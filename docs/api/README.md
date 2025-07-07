# API Documentation

This section contains the comprehensive API reference for the DiffEq library.

## 🔗 Auto-Generated Documentation

The complete API documentation is automatically generated using Doxygen and is available at:

- **[HTML Documentation](../generated/html/index.html)** - Complete API reference with inheritance diagrams
- **[XML Documentation](../generated/xml/)** - Machine-readable API documentation

## 📋 Quick Reference

### Core Concepts

- **State Types** - Supported state representations (vectors, arrays, custom types)
- **Integrator Interface** - Common interface for all integrators
- **Step Control** - Adaptive step size control mechanisms
- **Error Estimation** - Built-in error estimation and control

### Namespaces

- `diffeq::concepts` - C++20 concepts for type checking
- `diffeq::integrators::ode` - ODE integrators
- `diffeq::integrators::sde` - SDE integrators
- `diffeq::state` - State management utilities
- `diffeq::traits` - Type traits and metaprogramming utilities

### Main Classes

#### ODE Integrators

- `euler<StateType>` - Simple Euler method
- `improved_euler<StateType>` - Improved Euler (Heun's method)
- `rk4<StateType>` - Fourth-order Runge-Kutta
- `rk23<StateType>` - Adaptive Runge-Kutta 2(3)
- `rk45<StateType>` - Dormand-Prince method
- `dop853<StateType>` - High-order adaptive method
- `bdf<StateType>` - Backward differentiation formulas
- `lsoda<StateType>` - Automatic stiffness detection

#### SDE Integrators

- `euler_maruyama<StateType>` - Basic SDE integration
- `milstein<StateType>` - Higher-order SDE method
- `sri1<StateType>` - Stochastic Runge-Kutta
- `sra1<StateType>`, `sra2<StateType>` - Simplified stochastic RK
- `sosri<StateType>`, `sosra<StateType>` - Second-order methods
- `implicit_euler_maruyama<StateType>` - For stiff SDEs

## 🔧 Usage Patterns

### Basic Integration

```cpp
// Create integrator
diffeq::integrators::rk4<std::vector<double>> integrator;

// Define system
auto system = [](double t, const auto& y, auto& dydt) {
    dydt[0] = -0.1 * y[0];
};

// Integrate
std::vector<double> state = {1.0};
double time = 0.0;
double dt = 0.01;

integrator.step(system, state, time, dt);
```

### Adaptive Integration

```cpp
// Create adaptive integrator
diffeq::integrators::rk45<std::vector<double>> integrator;

// Set tolerances
integrator.set_tolerances(1e-8, 1e-6);

// Integrate with automatic step size control
integrator.integrate(system, state, t_start, t_end, dt_initial);
```

### Stochastic Integration

```cpp
// Create SDE integrator
diffeq::integrators::euler_maruyama<std::vector<double>> integrator;

// Define drift and diffusion
auto drift = [](double t, const auto& y, auto& dydt) {
    dydt[0] = -0.1 * y[0];
};

auto diffusion = [](double t, const auto& y, auto& dgdt) {
    dgdt[0] = 0.1;
};

// Integrate with noise
integrator.step(drift, diffusion, state, time, dt);
```

## 🎯 Advanced Features

### Parallel Execution

```cpp
// Use with standard library algorithms
std::vector<std::vector<double>> initial_conditions = /* ... */;

std::for_each(std::execution::par, 
              initial_conditions.begin(), 
              initial_conditions.end(),
              [&](auto& state) {
                  integrator.integrate(system, state, t0, t1, dt);
              });
```

### Custom State Types

```cpp
// Define custom state type
struct MyState {
    double x, y, z;
    
    // Required operations
    MyState operator+(const MyState& other) const;
    MyState operator*(double scalar) const;
    // ... other required operations
};

// Use with integrator
diffeq::integrators::rk4<MyState> integrator;
```

## 📚 Detailed Documentation

For complete details on all classes, methods, and advanced usage, please refer to the auto-generated Doxygen documentation.

## 🐛 Error Handling

The library uses exceptions for error handling:

- `std::invalid_argument` - Invalid parameters
- `std::runtime_error` - Runtime integration errors
- `std::out_of_range` - Array bounds errors

## 🔗 See Also

- [Examples](../examples/README.md) - Practical usage examples
- [Performance Guide](../performance/README.md) - Optimization tips
- [Main Documentation](../index.md) - Library overview 