# DiffEq - Pure C++ ODE Integration Library

A high-performance, header-only C++ library for solving ordinary differential equations (ODEs) with adaptive step size control.

## Key Features

- **Pure C++20** - No Python, SciPy, or external dependencies
- **Header-only** - Easy integration into your projects
- **High-accuracy integrators** - Including DOP853 (8th order Dormand-Prince)
- **Template-based design** - Works with various state types (vectors, Eigen arrays, etc.)
- **SciPy-consistent** - DOP853 implementation matches SciPy's behavior
- **Comprehensive testing** - Validated against reference solutions

## Supported Integrators

- **RK4**: Classic 4th order Runge-Kutta
- **RK23**: 2nd/3rd order adaptive Runge-Kutta
- **RK45**: 4th/5th order adaptive Runge-Kutta (Dormand-Prince)
- **DOP853**: 8th order adaptive Dormand-Prince (high accuracy)
- More integrators in development...

## Quick Example

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
    DOP853Integrator<std::vector<double>> integrator(exponential_decay);
    integrator.set_time(0.0);
    
    // Integrate from t=0 to t=1
    integrator.integrate(y, 0.01, 1.0);
    
    std::cout << "Solution at t=1: " << y[0] << std::endl;
    std::cout << "Exact solution: " << std::exp(-1.0) << std::endl;
    std::cout << "Error: " << std::abs(y[0] - std::exp(-1.0)) << std::endl;
    
    return 0;
}
```

## Building

### Prerequisites
- C++20 compatible compiler (GCC 10+, Clang 10+, MSVC 2019+)
- xmake build system

### Build and Test
```bash
cd diffeq
xmake          # Build all examples and tests
xmake test     # Run comprehensive test suite
```

## Usage

1. **Include the header**: `#include <diffeq.hpp>`
2. **Define your ODE system**: Function that computes dy/dt = f(t, y)
3. **Choose an integrator**: RK4, RK45, DOP853, etc.
4. **Set initial conditions and integrate**

See `examples/` directory for more detailed examples.

## Documentation

- [`DOP853_QUICK_START.md`](DOP853_QUICK_START.md) - Quick start guide for the high-accuracy DOP853 integrator
- [`DOP853_IMPLEMENTATION_SUMMARY.md`](DOP853_IMPLEMENTATION_SUMMARY.md) - Technical details and validation
- [`examples/`](examples/) - Example usage for different integrators
- [`include/`](include/) - Header files with inline documentation

## Pure C++ Design

This library is designed to be completely self-contained:
- **No runtime dependencies** beyond the C++ standard library
- **No Python or SciPy required** for building, testing, or usage
- **No external libraries** required (optional integration with Eigen available)
- **Header-only implementation** for easy deployment

Development tools (like reference data generators) are kept separate in the `dev/` folder.

## License

See [`LICENSE`](LICENSE) file for details.

## Contributing

Contributions welcome! Please ensure all tests pass and maintain the pure C++ design philosophy.
