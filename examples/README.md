# diffeq Examples

This directory contains comprehensive examples demonstrating how to use the diffeq library for solving differential equations.

## Example Programs

### Core Integration Examples

- **`working_integrators_demo.cpp`** - Demonstrates all working ODE integrators (RK4, RK23, RK45, BDF, LSODA)
- **`rk4_integrator_usage.cpp`** - Basic RK4 integrator usage with various ODE systems
- **`advanced_integrators_usage.cpp`** - Advanced integrator features and configurations
- **`state_concept_usage.cpp`** - Shows how to use different state types (vectors, arrays, custom types)
- **`std_async_integration_demo.cpp`** - Direct use of C++ standard library async facilities without unnecessary abstractions
- **`coroutine_integration_demo.cpp`** - C++20 coroutines integration for fine-grained execution control and cooperative multitasking
- **`timeout_integration_demo.cpp`** - Timeout-protected integration for robust applications
- **`seamless_parallel_timeout_demo.cpp`** - Seamless integration of timeout + async + parallel execution

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
- **`coroutine_integration_demo.cpp`** - C++20 Coroutines integration:
  - Fine-grained execution control with pause/resume
  - Cooperative multitasking for multiple integrations
  - Zero-overhead state preservation between yields
  - Progress monitoring with minimal overhead
  - Interruptible long-running computations
- **`advanced_gpu_async_demo.cpp`** - GPU acceleration with async processing
- **`realtime_signal_processing.cpp`** - Real-time signal processing integration
- **`composable_facilities_demo.cpp`** 🎯 **NEW: Solves Combinatorial Explosion** - Composable architecture demonstration:
  - High cohesion, low coupling design principles
  - Independent facilities: Timeout, Parallel, Async, Signals, Output
  - Flexible composition using decorator pattern
  - Order-independent facility stacking
  - Linear scaling (N classes for N facilities, not 2^N)
  - Extensibility without modifying existing code
  - Real-world usage scenarios and performance analysis

### Testing and Validation

- **`quick_test.cpp`** - Quick validation tests
- **`test_dop853.cpp`** - DOP853 integrator testing
- **`test_rk4_only.cpp`** - RK4 integrator testing
- **`sde_demo.cpp`** - Basic SDE demonstration

## Building and Running Examples

### Prerequisites

- C++17 or later compiler
- xmake build system
- Optional: OpenMP, Intel TBB, CUDA for advanced parallelism examples

### Building

```bash
# Using xmake
xmake
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

### 4. Modern C++ Features
- **C++20 Coroutines**: Fine-grained control and cooperative multitasking in `coroutine_integration_demo.cpp`
- **Standard Library Async**: Direct use of std::async without abstractions in `std_async_integration_demo.cpp`
- **Parallel Execution**: Hardware-optimized parallel processing in `parallelism_usage_demo.cpp`

### 5. Domain-Specific Examples
- **Finance**: Black-Scholes, Heston models in `sde_usage_demo.cpp`
- **Robotics**: Control systems in `parallelism_usage_demo.cpp`
- **Scientific**: Chemical reactions, ecosystem models in `sde_usage_demo.cpp`

## Key Features Demonstrated

### Integration Methods
- **ODE Solvers**: RK4, RK23, RK45, BDF, LSODA, DOP853
- **SDE Solvers**: Euler-Maruyama, Milstein, SRA1, SOSRA, SRIW1, SOSRI
- **Adaptive Methods**: Automatic step size control
- **Stiff Systems**: BDF and LSODA for stiff problems

### Modern C++ Features
- **C++20 Coroutines**: 
  - Pausable/resumable integration with `co_yield`
  - Fine-grained CPU control for real-time systems
  - Zero-overhead state preservation
  - Cooperative multitasking between multiple integrations
  - Progress monitoring without blocking the main thread

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
- **Timeout Protection**: Prevents hanging integrations with configurable timeouts
- **Progress Monitoring**: Real-time integration progress tracking and cancellation
- **Seamless Parallelization**: Automatic hardware utilization without configuration
- **Execution Strategy Selection**: Auto-chooses optimal approach based on problem and hardware

## Best Practices

1. **Start Simple**: Begin with `working_integrators_demo.cpp` to understand basic usage
2. **Choose the Right Integrator**: Use RK45 for general problems, BDF for stiff systems
3. **Leverage Auto-Optimization**: Use `diffeq::integrate_auto()` for automatic hardware utilization
4. **Handle Real-time Requirements**: Use interface examples for systems with external signals
5. **Use Timeout Protection**: Add timeout protection for production applications
6. **Scale Seamlessly**: From single integration to batch processing with `seamless_parallel_timeout_demo.cpp`
7. **Compose Facilities**: Use the composable architecture for flexible combinations of capabilities
   - Start with `make_builder(base_integrator)`
   - Add only the facilities you need: `.with_timeout()`, `.with_parallel()`, etc.
   - Avoid combinatorial explosion - compose instead of inheriting
   - Order doesn't matter - decorators work in any sequence
8. **Validate Results**: Compare with analytical solutions when available

## Troubleshooting

### Common Issues
- **Compilation Errors**: Ensure C++17 support and required libraries
- **Performance Issues**: Check parallel backend availability
- **Accuracy Problems**: Verify integrator choice and tolerances
- **Memory Issues**: Use appropriate state types and batch sizes
- **Hanging Integration**: Use timeout protection for robust applications

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