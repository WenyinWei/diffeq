# diffeq Examples

This directory contains comprehensive examples demonstrating how to use the diffeq library for solving differential equations.

## Example Programs

### Core Integration Examples

- **`working_integrators_demo.cpp`** - Demonstrates all working ODE integrators (RK4, RK23, RK45, BDF, LSODA)
- **`rk4_integrator_usage.cpp`** - Basic RK4 integrator usage with various ODE systems
- **`advanced_integrators_usage.cpp`** - Advanced integrator features and configurations
- **`state_concept_usage.cpp`** - Shows how to use different state types (vectors, arrays, custom types)

### Parallelism Examples

- **`parallelism_usage_demo.cpp`** - Comprehensive parallelism features including:
  - Quick start parallel interface
  - Robotics control systems with real-time parallelism
  - Stochastic process research with GPU-accelerated Monte Carlo
  - Multi-hardware target benchmarking
- **`standard_parallelism_demo.cpp`** - Standard library parallelism integration:
  - C++17/20 std::execution policies
  - OpenMP parallel loops
  - Intel TBB integration
  - Task-based async dispatchers
- **`simple_standard_parallelism.cpp`** - Simplified parallel usage patterns
- **`standard_parallelism_demo.cpp`** - Advanced standard parallelism features
- **`simplified_parallel_usage.cpp`** - Easy-to-use parallel interfaces
- **`test_advanced_parallelism.cpp`** - Testing advanced parallelism features

### Advanced Features

- **`interface_usage_demo.cpp`** - Integration interface examples:
  - Financial portfolio modeling with signal processing
  - Robotics control with real-time feedback
  - Scientific simulations with parameter updates
- **`sde_usage_demo.cpp`** - Stochastic Differential Equation examples:
  - Black-Scholes financial models
  - Heston stochastic volatility
  - Noisy oscillator control systems
  - Stochastic Lotka-Volterra ecosystem models
- **`advanced_gpu_async_demo.cpp`** - GPU acceleration with async processing
- **`realtime_signal_processing.cpp`** - Real-time signal processing integration

### Testing and Validation

- **`quick_test.cpp`** - Quick validation tests
- **`test_dop853.cpp`** - DOP853 integrator testing
- **`test_rk4_only.cpp`** - RK4 integrator testing
- **`sde_demo.cpp`** - Basic SDE demonstration

## Building and Running Examples

### Prerequisites

- C++17 or later compiler
- CMake or xmake build system
- Optional: OpenMP, Intel TBB, CUDA for advanced parallelism examples

### Building

```bash
# Using xmake (recommended)
xmake

# Or using CMake
mkdir build && cd build
cmake ..
make
```

### Running Examples

```bash
# Run a specific example
./examples/working_integrators_demo

# Run parallelism examples
./examples/parallelism_usage_demo
./examples/standard_parallelism_demo

# Run SDE examples
./examples/sde_usage_demo

# Run interface examples
./examples/interface_usage_demo
```

## Example Categories

### 1. Basic Usage
Start with these examples to understand the fundamentals:
- `working_integrators_demo.cpp`
- `rk4_integrator_usage.cpp`
- `state_concept_usage.cpp`

### 2. Parallelism
For performance-critical applications:
- `parallelism_usage_demo.cpp` - Full-featured parallelism
- `standard_parallelism_demo.cpp` - Standard library integration
- `simple_standard_parallelism.cpp` - Easy parallel usage

### 3. Advanced Features
For complex applications:
- `interface_usage_demo.cpp` - Signal processing and real-time integration
- `sde_usage_demo.cpp` - Stochastic differential equations
- `advanced_gpu_async_demo.cpp` - GPU acceleration

### 4. Domain-Specific Examples
- **Finance**: Black-Scholes, Heston models in `sde_usage_demo.cpp`
- **Robotics**: Control systems in `parallelism_usage_demo.cpp`
- **Scientific**: Chemical reactions, ecosystem models in `sde_usage_demo.cpp`

## Key Features Demonstrated

### Integration Methods
- **ODE Solvers**: RK4, RK23, RK45, BDF, LSODA, DOP853
- **SDE Solvers**: Euler-Maruyama, Milstein, SRA1, SOSRA, SRIW1, SOSRI
- **Adaptive Methods**: Automatic step size control
- **Stiff Systems**: BDF and LSODA for stiff problems

### Parallelism
- **CPU Parallelism**: std::execution, OpenMP, Intel TBB
- **GPU Acceleration**: CUDA, Thrust integration
- **Async Processing**: Task-based parallel execution
- **Real-time Control**: Low-latency parallel integration

### Advanced Features
- **Signal Processing**: Real-time event handling
- **Parameter Sweeps**: Parallel parameter studies
- **Multi-physics**: Coupled system integration
- **Hardware Optimization**: Automatic backend selection

## Best Practices

1. **Start Simple**: Begin with `working_integrators_demo.cpp` to understand basic usage
2. **Choose the Right Integrator**: Use RK45 for general problems, BDF for stiff systems
3. **Leverage Parallelism**: Use parallel examples for performance-critical applications
4. **Handle Real-time Requirements**: Use interface examples for systems with external signals
5. **Validate Results**: Compare with analytical solutions when available

## Troubleshooting

### Common Issues
- **Compilation Errors**: Ensure C++17 support and required libraries
- **Performance Issues**: Check parallel backend availability
- **Accuracy Problems**: Verify integrator choice and tolerances
- **Memory Issues**: Use appropriate state types and batch sizes

### Getting Help
- Check the main library documentation
- Review the test suite for usage patterns
- Examine the source code for implementation details

## Contributing

When adding new examples:
1. Follow the existing naming convention
2. Include comprehensive comments
3. Demonstrate realistic use cases
4. Add to this README if appropriate
5. Ensure the example compiles and runs correctly 