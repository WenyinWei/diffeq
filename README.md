# DiffEq - Modern C++ ODE Integration Library with Async Signal Processing

A high-performance, header-only C++ library for solving ordinary differential equations (ODEs) with advanced async execution and signal processing capabilities for financial and robotics applications. **Now fully modernized with standard C++20/23 facilities only.**

## 🚀 Key Features

- **Pure C++20/23** - Modern C++ with no external dependencies
- **Header-only** - Easy integration into your projects  
- **Async Integration** - Non-blocking ODE solving with std::future and std::thread
- **Signal Processing** - Type-safe event handling and real-time data processing
- **Domain-specific Processors** - Built-in support for finance and robotics applications
- **High-accuracy integrators** - Including DOP853 (8th order Dormand-Prince)
- **Template-based design** - Works with various state types (vectors, arrays, etc.)
- **SciPy-consistent** - Core integrators match SciPy's behavior
- **Standard C++ Only** - No platform-specific code or custom IPC systems
- **Future-ready** - Designed for potential C++ standardization

## ✨ What's New in the Modernized Version

- **Removed custom communication system** - Replaced with standard C++ async facilities
- **New async module** - `include/async/async_integrator.hpp` with std::future-based integration
- **New signal processing** - `include/signal/signal_processor.hpp` for type-safe event handling
- **Modern CMake/Xmake** - Updated build system for C++20 header-only library
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

### Async Integration with Signal Processing
```cpp
#include <diffeq.hpp>
#include <async/async_integrator.hpp>
#include <signal/signal_processor.hpp>
#include <vector>
#include <future>

using namespace diffeq;

int main() {
    std::vector<double> state = {1.0, 0.0};  // Initial conditions
    
    // Create async integrator
    auto integrator = async::factory::make_async_rk45<std::vector<double>>(
        [](double t, const auto& y, auto& dydt) {
            dydt[0] = y[1];      // dx/dt = v
            dydt[1] = -y[0];     // dv/dt = -x (harmonic oscillator)
        },
        async::AsyncIntegrator<std::vector<double>>::Config{
            .enable_async_stepping = true,
            .enable_state_monitoring = true
        }
    );
    
    // Create signal processor for real-time data handling
    auto signal_proc = signal::make_signal_processor(integrator);
    
    // Register signal handler for external events
    signal_proc->register_handler<double>("market_update", 
        [&](const signal::Signal<double>& sig) {
            std::cout << "Received market data: " << sig.data << std::endl;
        });
    
    // Start async integration
    auto future = integrator->integrate_async(state, 0.01, 10.0);
    
    // Simulate external signals
    signal_proc->emit_signal("market_update", 42.5);
    
    // Wait for completion
    auto result = future.get();
    std::cout << "Final state: [" << result[0] << ", " << result[1] << "]" << std::endl;
    
    return 0;
}
```

### Domain-Specific Application (Finance)
```cpp
#include <diffeq.hpp>
#include <domains/application_processors.hpp>
#include <vector>

using namespace diffeq::domains;

int main() {
    // Portfolio state: [asset1, asset2, asset3, VaR, sharpe_ratio]
    std::vector<double> portfolio = {100000.0, 150000.0, 120000.0, 5000.0, 1.2};
    
    // Create finance processor
    auto finance_proc = factory::make_finance_processor();
    
    // Set up portfolio dynamics
    auto integrator = diffeq::make_rk45<std::vector<double>>(
        [&](double t, const auto& y, auto& dydt) {
            finance_proc->portfolio_dynamics(t, y, dydt);
        });
    
    // Real-time portfolio optimization
    integrator.integrate(portfolio, 0.001, 1.0);  // 1 day simulation
    
    std::cout << "Optimized portfolio value: " 
              << (portfolio[0] + portfolio[1] + portfolio[2]) << std::endl;
    
    return 0;
}
```
    
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

The modernized diffeq library follows a clean, modular architecture:

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
├── async/                         # NEW: Async execution (standard C++ only)
│   └── async_integrator.hpp      # std::future-based async integration
├── signal/                        # NEW: Signal processing 
│   └── signal_processor.hpp      # Type-safe event handling
├── domains/                       # NEW: Domain-specific processors
│   └── application_processors.hpp # Finance and robotics
└── communication/                 # DEPRECATED: Custom IPC (use async/ instead)
    ├── process_connector.hpp      # [DEPRECATED]
    ├── event_bus.hpp             # [DEPRECATED]  
    └── realtime_priority.hpp     # [DEPRECATED]
```

## Migration Guide

If you were using the old communication system, here's how to migrate:

### Old Communication System (Deprecated)
```cpp
// OLD: Don't use this anymore
#include <communication/process_connector.hpp>
#include <communication/event_bus.hpp>

ProcessConnector connector;
EventBus event_bus;
// Complex IPC setup...
```

### New Async/Signal System (Recommended)
```cpp
// NEW: Use this instead
#include <async/async_integrator.hpp>
#include <signal/signal_processor.hpp>

auto integrator = async::factory::make_async_rk45<StateType>(ode_system);
auto signal_proc = signal::make_signal_processor(integrator);
// Simple, standard C++ only
```

### Benefits of Migration
- **Standard C++ only** - No platform-specific code
- **Better performance** - Less overhead than custom IPC
- **Type safety** - Template-based instead of std::any
- **Maintainability** - Cleaner separation of concerns
- **Future-proof** - Suitable for C++ standardization

## Usage

1. **Include the main header**: `#include <diffeq.hpp>`
2. **Define your ODE system**: Function that computes dy/dt = f(t, y)
3. **Choose an integrator**: Use factory functions like `make_rk45()`, `make_dop853()`
4. **Set initial conditions and integrate**
5. **Optional**: Add async execution and signal processing for real-time applications

### Integrator Selection Guide

- **`make_rk45()`**: Best general-purpose choice (Dormand-Prince 5th order)
- **`make_dop853()`**: High-accuracy applications (8th order)  
- **`make_bdf()`**: Stiff systems (backward differentiation)
- **`make_lsoda()`**: Automatic stiffness detection

See `examples/` directory for detailed usage examples.

## Advanced Features

### Async Integration
```cpp
#include <async/async_integrator.hpp>

auto integrator = async::factory::make_async_dop853<StateType>(ode_system);
auto future = integrator->integrate_async(initial_state, dt, t_final);
// Continue other work...
auto result = future.get();  // Get result when ready
```

### Signal Processing
```cpp  
#include <signal/signal_processor.hpp>

auto signal_proc = signal::make_signal_processor(integrator);
signal_proc->register_handler<MarketData>("price_update", handle_price_change);
signal_proc->emit_signal("price_update", market_data);
```

### Domain-Specific Applications
```cpp
#include <domains/application_processors.hpp>

// Finance: Portfolio optimization
auto finance_proc = domains::factory::make_finance_processor();

// Robotics: Robot control  
auto robotics_proc = domains::factory::make_robotics_processor<6>(); // 6-DOF robot
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
