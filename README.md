# DiffEq - Modern C++ ODE Integration Library with Async Signal Processing

A high-performance, header-only C++ library for solving ordinary differential equations (ODEs) with advanced async execution and signal processing capabilities for financial and robotics applications. **Now fully modernized with standard C++20/23 facilities only.**

## 🚀 Key Features

- **Pure C++20/23** - Modern C++ with no external dependencies
- **Header-only** - Easy integration into your projects  
- **Unified Interface** - Single extensible interface for signal-aware ODE integration
- **Signal Processing** - Type-safe event handling and real-time data processing
- **Cross-domain Support** - Works for finance, robotics, science, and any application domain
- **High-accuracy integrators** - Including DOP853 (8th order Dormand-Prince)
- **Template-based design** - Works with various state types using C++ concepts
- **SciPy-consistent** - Core integrators match SciPy's behavior
- **Standard C++ Only** - No platform-specific code or custom systems
- **Future-ready** - Designed for potential C++ standardization

## ✨ What's New in the Modernized Version

- **Unified Interface** - Single `IntegrationInterface` applies to domains such as robotics control, financial exchange, and so on.
- **Proper C++ concepts** - Template syntax uses `system_state` and `can_be_time` concepts
- **Removed communication module** - Replaced with standard C++ async facilities
- **Signal-aware integration** - Built-in support for discrete events, continuous influences, and real-time output
- **Extensible design** - Works for any application domain with the same unified interface
- **Clean architecture** - Better separation of concerns and modularity

## Supported Integrators

### Fixed Step Methods
- **EulerIntegrator**: Simple 1st order explicit method
- **ImprovedEulerIntegrator**: 2nd order Heun's method  
- **RK4Integrator**: Classic 4th order Runge-Kutta

### Adaptive Step Methods (Non-stiff)
- **RK23Integrator**: 3rd order Bogacki-Shampine with error control
- **RK45Integrator**: 5th order Dormand-Prince (recommended default)
- **DOP853Integrator**: 8th order high-accuracy method

### Stiff System Methods
- **RadauIntegrator**: 5th order implicit Runge-Kutta
- **BDFIntegrator**: Variable order (1-6) backward differentiation

### Stochastic Differential Equations (SDEs)

#### Basic SDE Methods
- **EulerMaruyamaIntegrator**: Basic SDE solver (strong order 0.5)
- **MilsteinIntegrator**: Higher-order method with Lévy area (strong order 1.0)
- **SRI1Integrator**: Stochastic Runge-Kutta method (strong order 1.0)
- **ImplicitEulerMaruyamaIntegrator**: Implicit method for stiff SDEs

#### Advanced High-Order Methods (Strong Order 1.5)
- **SRAIntegrator**: Stochastic Runge-Kutta for additive noise SDEs
- **SRIIntegrator**: Stochastic Runge-Kutta for general Itô SDEs  
- **SOSRAIntegrator**: Stability-optimized SRA, robust to stiffness
- **SOSRIIntegrator**: Stability-optimized SRI, robust to stiffness

#### Pre-configured Methods
- **SRA1, SRA2**: Different tableau variants for additive noise
- **SRIW1**: Weak order 2.0 variant for general SDEs
- **SOSRA, SOSRI**: Stability-optimized for high tolerances

*All SDE methods are inspired by the DifferentialEquations.jl/StochasticDiffEq.jl algorithms with proper tableau-based implementations.*

### Automatic Methods  
- **LSODAIntegrator**: Automatic switching between Adams and BDF

## Quick Examples

### Basic ODE Integration
```cpp
#include <diffeq.hpp>
#include <vector>
#include <iostream>

// Define your ODE: dy/dt = -y (exponential decay)
void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

int main() {
    std::vector<double> y = {1.0};  // Initial condition: y(0) = 1
    
    // Create high-accuracy DOP853 integrator  
    auto integrator = diffeq::make_dop853<std::vector<double>>(exponential_decay);
    
    // Integrate from t=0 to t=1
    integrator.integrate(y, 0.01, 1.0);
    
    std::cout << "Solution at t=1: " << y[0] << std::endl;
    std::cout << "Exact solution: " << std::exp(-1.0) << std::endl;
    std::cout << "Error: " << std::abs(y[0] - std::exp(-1.0)) << std::endl;
    
    return 0;
}
```

### Signal-Aware Integration with Unified Interface
```cpp
#include <diffeq.hpp>
#include <interfaces/integration_interface.hpp>

using namespace diffeq;

// Create unified interface (works for ANY domain)
auto interface = interfaces::make_integration_interface<std::vector<double>, double>();

// Register signal influences for real-time events
interface->register_signal_influence<double>("price_update",
    interfaces::IntegrationInterface<std::vector<double>, double>::InfluenceMode::CONTINUOUS_SHIFT,
    [](const double& price, auto& state, auto t) {
        // Modify portfolio dynamics based on price signal
        double momentum = (price > 100.0) ? 0.01 : -0.01;
        for (auto& asset : state) asset *= (1.0 + momentum);
    });

// Register real-time output monitoring
interface->register_output_stream("portfolio_monitor",
    [](const auto& state, auto t) {
        std::cout << "Portfolio value: " << std::accumulate(state.begin(), state.end(), 0.0) << std::endl;
    });

// Create signal-aware ODE (combines your ODE with signal processing)
auto signal_ode = interface->make_signal_aware_ode(my_portfolio_ode);
auto integrator = make_rk45<std::vector<double>>(signal_ode);

// Integration automatically handles signals and outputs
std::vector<double> state = {100000.0, 150000.0, 120000.0}; // Initial portfolio
integrator.integrate(state, 0.001, 1.0); // 1 day simulation
```

### Cross-Domain Applications (Finance, Robotics, Science)
```cpp
#include <diffeq.hpp>
#include <interfaces/integration_interface.hpp>

using namespace diffeq;

int main() {
    // FINANCE: Portfolio optimization with the unified interface
    auto finance_interface = interfaces::make_integration_interface<std::vector<double>, double>();
    
    finance_interface->register_signal_influence<double>("market_volatility",
        interfaces::IntegrationInterface<std::vector<double>, double>::InfluenceMode::DISCRETE_EVENT,
        [](const double& volatility, auto& portfolio, auto t) {
            if (volatility > 0.3) {
                // Reduce risk exposure
                for (auto& asset : portfolio) asset *= 0.9;
            }
        });
    
    // ROBOTICS: Multi-joint robot control with the same unified interface
    auto robot_interface = interfaces::make_integration_interface<std::array<double, 18>, double>();
    
    robot_interface->register_signal_influence<std::array<double, 6>>("joint_targets",
        interfaces::IntegrationInterface<std::array<double, 18>, double>::InfluenceMode::CONTINUOUS_SHIFT,
        [](const auto& targets, auto& robot_state, auto t) {
            // Update control targets for 6-DOF robot
            for (size_t i = 0; i < targets.size(); ++i) {
                robot_state[i + 12] = targets[i]; // Set target positions
            }
        });
    
    // Emergency stop capability
    robot_interface->register_signal_influence<bool>("emergency_stop",
        interfaces::IntegrationInterface<std::array<double, 18>, double>::InfluenceMode::DISCRETE_EVENT,
        [](bool stop, auto& robot_state, auto t) {
            if (stop) {
                // Zero all velocities immediately
                for (size_t i = 6; i < 12; ++i) robot_state[i] = 0.0;
            }
        });
    
    // Both finance and robotics use the SAME unified interface!
    // No domain-specific processors needed anymore.
    
    return 0;
}
```

### Stochastic Differential Equations (SDE) 
```cpp
#include <diffeq.hpp>
#include <sde/sde_solvers.hpp>
#include <sde/advanced_sde_solvers.hpp>

using namespace diffeq::sde;

// Define Black-Scholes model: dS = μS dt + σS dW
void black_scholes_drift(double t, const std::vector<double>& S, std::vector<double>& dS) {
    double mu = 0.05;  // Expected return
    dS[0] = mu * S[0];
}

void black_scholes_diffusion(double t, const std::vector<double>& S, std::vector<double>& gS) {
    double sigma = 0.2;  // Volatility
    gS[0] = sigma * S[0];
}

int main() {
    // Create SDE problem
    auto problem = factory::make_sde_problem<std::vector<double>, double>(
        black_scholes_drift, black_scholes_diffusion, NoiseType::DIAGONAL_NOISE);
    
    auto wiener = factory::make_wiener_process<std::vector<double>, double>(1, 12345);
    
    std::vector<double> S = {100.0};  // Initial stock price
    
    // Compare different SDE methods
    
    // Basic Euler-Maruyama (strong order 0.5)
    {
        EulerMaruyamaIntegrator<std::vector<double>, double> integrator(problem, wiener);
        std::vector<double> price = S;
        integrator.integrate(price, 0.01, 1.0);
        std::cout << "Euler-Maruyama: " << price[0] << std::endl;
    }
    
    // Advanced SRA1 (strong order 1.5 for additive-like noise)
    {
        auto integrator = factory::make_sra1_integrator<std::vector<double>, double>(problem, wiener);
        std::vector<double> price = S;
        integrator->integrate(price, 0.01, 1.0);
        std::cout << "SRA1: " << price[0] << std::endl;
    }
    
    // Stability-optimized SOSRI (robust for stiff problems)
    {
        auto integrator = factory::make_sosri_integrator<std::vector<double>, double>(problem, wiener);
        std::vector<double> price = S;
        integrator->integrate(price, 0.01, 1.0);
        std::cout << "SOSRI: " << price[0] << std::endl;
    }
    
    // Multi-dimensional SDE: Heston stochastic volatility
    auto heston_drift = [](double t, const std::vector<double>& x, std::vector<double>& dx) {
        double S = x[0], V = x[1];
        double mu = 0.05, kappa = 2.0, theta = 0.04;
        dx[0] = mu * S;
        dx[1] = kappa * (theta - V);
    };
    
    auto heston_diffusion = [](double t, const std::vector<double>& x, std::vector<double>& gx) {
        double S = x[0], V = std::max(x[1], 0.0);
        double sigma = 0.3;
        gx[0] = std::sqrt(V) * S;
        gx[1] = sigma * std::sqrt(V);
    };
    
    auto heston_problem = factory::make_sde_problem<std::vector<double>, double>(
        heston_drift, heston_diffusion, NoiseType::GENERAL_NOISE);
    
    auto heston_wiener = factory::make_wiener_process<std::vector<double>, double>(2, 54321);
    
    // Use SOSRA for robust performance on this complex model
    auto heston_integrator = factory::make_sosra_integrator<std::vector<double>, double>(
        heston_problem, heston_wiener);
    
    std::vector<double> heston_state = {100.0, 0.04};  // [S, V]
    heston_integrator->integrate(heston_state, 0.01, 1.0);
    
    std::cout << "Heston model: S = " << heston_state[0] 
              << ", V = " << heston_state[1] << std::endl;
    
    return 0;
}
```

## Building

### Prerequisites
- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake 3.20+ (recommended) or xmake

### Modern CMake Build (Recommended)
```bash
cd diffeq
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest  # Run tests
```

### Legacy xmake Build
```bash
cd diffeq
xmake          # Build all examples and tests
xmake test     # Run comprehensive test suite
```

### Header-Only Usage
Since the library is header-only, you can also simply:
```cpp
#include "path/to/diffeq/include/diffeq.hpp"
```

Or with CMake:
```cmake
find_package(diffeq REQUIRED)
target_link_libraries(your_target diffeq::diffeq)
```

## Architecture Overview

The modernized diffeq library follows a clean, unified architecture:

```
include/diffeq.hpp                 # Main header (includes everything)
├── core/                          # Core ODE concepts and base classes
│   ├── concepts.hpp              # C++20 concepts for type safety
│   ├── abstract_integrator.hpp   # Base integrator interface
│   └── adaptive_integrator.hpp   # Adaptive step size base
├── solvers/                       # All integrator implementations
│   ├── rk45_solver.hpp           # Dormand-Prince (recommended)
│   ├── dop853_solver.hpp         # High-accuracy 8th order
│   ├── bdf_solver.hpp            # For stiff systems
│   └── ...                       # Other solvers
├── interfaces/                    # NEW: Unified interface (replaces domains)
│   └── integration_interface.hpp # Single interface for all applications
├── async/                         # Async execution (standard C++ only)
│   └── async_integrator.hpp      # std::future-based async integration
├── signal/                        # Signal processing 
│   └── signal_processor.hpp      # Type-safe event handling
└── examples/                      # Usage examples
    └── interface_usage.hpp        # How to use the unified interface
```

### Key Architectural Changes

- **Unified Interface**: Single `IntegrationInterface` replaces all domain-specific processors
- **Removed Legacy Code**: No more `communication/` or `domains/` directories
- **C++ Concepts**: Proper template constraints using `system_state` and `can_be_time`
- **Standard C++ Only**: All functionality using C++20/23 standard library

## Testing

The library includes comprehensive tests in the `test/` directory:

```bash
# Compile and run integration tests
cd diffeq
g++ -std=c++20 -I include test/integration/test_modernized_interface.cpp -o test_modernized
./test_modernized

# Compile and run unit tests
g++ -std=c++20 -I include test/unit/test_*.cpp -o test_units
./test_units
```

The main integration test (`test/integration/test_modernized_interface.cpp`) validates:
- Unified interface functionality across all domains
- Signal processing and event handling
- C++ concepts compliance
- Async integration capabilities
- Cross-domain usage patterns (finance, robotics, science)

## Usage

1. **Include the main header**: `#include <diffeq.hpp>`
2. **Define your ODE system**: Function that computes dy/dt = f(t, y)
3. **Choose an integrator**: Use factory functions like `make_rk45()`, `make_dop853()`
4. **Set initial conditions and integrate**
5. **Optional**: Use unified interface for signal-aware integration across any domain

### Basic Integration
```cpp
#include <diffeq.hpp>
auto integrator = diffeq::make_rk45<std::vector<double>>(my_ode);
integrator.integrate(state, dt, t_final);
```

### Signal-Aware Integration (Any Domain)
```cpp
#include <interfaces/integration_interface.hpp>
auto interface = diffeq::interfaces::make_integration_interface<StateType, TimeType>();
interface->register_signal_influence<DataType>("signal_name", mode, handler);
auto signal_ode = interface->make_signal_aware_ode(my_ode);
auto integrator = diffeq::make_rk45<StateType>(signal_ode);
```

### Integrator Selection Guide

- **`make_rk45()`**: Best general-purpose choice (Dormand-Prince 5th order)
- **`make_dop853()`**: High-accuracy applications (8th order)  
- **`make_bdf()`**: Stiff systems (backward differentiation)
- **`make_lsoda()`**: Automatic stiffness detection

See `examples/` directory for detailed usage examples.

## Advanced Features

### Unified Signal-Aware Integration
```cpp
#include <interfaces/integration_interface.hpp>

auto interface = interfaces::make_integration_interface<StateType, TimeType>();

// Discrete events (instantaneous state changes)
interface->register_signal_influence<EventType>("event_name",
    IntegrationInterface<StateType, TimeType>::InfluenceMode::DISCRETE_EVENT, handler);

// Continuous influences (modify ODE trajectory)  
interface->register_signal_influence<DataType>("signal_name",
    IntegrationInterface<StateType, TimeType>::InfluenceMode::CONTINUOUS_SHIFT, handler);

// Real-time output streams
interface->register_output_stream("stream_name", output_handler, interval);

auto signal_ode = interface->make_signal_aware_ode(original_ode);
```

### Async Integration
```cpp  
#include <async/async_integrator.hpp>

auto integrator = async::factory::make_async_dop853<StateType>(ode_system);
auto future = integrator->integrate_async(initial_state, dt, t_final);
// Continue other work...
future.wait(); // Wait for completion
```

### Cross-Domain Applications
```cpp
// Finance, robotics, science - all use the same unified interface!
auto finance_interface = interfaces::make_integration_interface<std::vector<double>, double>();
auto robot_interface = interfaces::make_integration_interface<std::array<double, 18>, double>();
auto science_interface = interfaces::make_integration_interface<std::vector<float>, float>();
```

## Documentation

- **Examples**: [`examples/`](examples/) - Complete usage examples
- **API Reference**: [`include/`](include/) - Header files with inline documentation  
- **Quick Start**: See examples above and in [`examples/realtime_signal_processing.cpp`](examples/realtime_signal_processing.cpp)
- **Migration Guide**: See "Migration Guide" section above

## Design Philosophy

This library prioritizes:

- **Standard C++ compliance** - Only C++20/23 standard features
- **Header-only design** - Easy integration and deployment
- **Zero external dependencies** - Self-contained and portable
- **Performance** - Optimized for high-frequency applications  
- **Type safety** - C++20 concepts and strong typing
- **Modularity** - Clean separation of concerns
- **Future-ready** - Designed for potential standardization

## Third-party Integration

While the library is self-contained, it integrates well with:

- **Networking**: Use Boost.Asio or standalone asio for advanced networking
- **JSON**: Use nlohmann/json for configuration and data exchange  
- **Linear Algebra**: Works with Eigen, Armadillo, or std::valarray
- **Plotting**: Output data to matplotlib-cpp, gnuplot, or CSV files

## License

See [`LICENSE`](LICENSE) file for details.

## Contributing

Contributions welcome! Please ensure all tests pass and maintain the pure C++ design philosophy.
