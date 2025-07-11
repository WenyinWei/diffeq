# Composable Integration Architecture

## Design Philosophy: High Cohesion, Low Coupling

The DiffEq library employs a **composable architecture** based on the decorator pattern to solve the combinatorial explosion problem when combining multiple facilities. This design achieves:

- **High Cohesion**: Each facility focuses on a single, well-defined concern
- **Low Coupling**: Facilities can be combined flexibly without tight dependencies
- **Linear Scaling**: Adding N facilities requires N classes, not 2^N classes

## The Problem

Without composable design, combining facilities leads to exponential class growth:

```cpp
// BAD: Combinatorial explosion
class TimeoutIntegrator { ... };
class ParallelIntegrator { ... };
class AsyncIntegrator { ... };
class SignalIntegrator { ... };
class OutputIntegrator { ... };

// Need all combinations:
class TimeoutParallelIntegrator { ... };
class TimeoutAsyncIntegrator { ... };
class TimeoutSignalIntegrator { ... };
class ParallelAsyncIntegrator { ... };
class ParallelSignalIntegrator { ... };
class AsyncSignalIntegrator { ... };
class TimeoutParallelAsyncIntegrator { ... };
class TimeoutParallelSignalIntegrator { ... };
class TimeoutAsyncSignalIntegrator { ... };
class ParallelAsyncSignalIntegrator { ... };
class TimeoutParallelAsyncSignalIntegrator { ... };
class TimeoutParallelAsyncSignalOutputIntegrator { ... };
// ... and so on (2^N classes for N facilities)
```

## The Solution: Decorator Pattern

Instead, we use independent decorators that can be composed:

```cpp
// GOOD: Composable decorators
auto integrator = make_builder(base_integrator)
    .with_timeout()
    .with_parallel()
    .with_signals()
    .with_output()
    .build();
```

## Architecture Components

### 1. Base Decorator Interface

All facilities inherit from `IntegratorDecorator<S, T>`:

```cpp
template<system_state S, can_be_time T = double>
class IntegratorDecorator : public AbstractIntegrator<S, T> {
protected:
    std::unique_ptr<AbstractIntegrator<S, T>> wrapped_integrator_;
public:
    // Delegates to wrapped integrator by default
    // Individual decorators override specific methods
};
```

### 2. Individual Facilities (High Cohesion)

Each decorator focuses on one concern:

#### Timeout Facility
```cpp
class TimeoutDecorator : public IntegratorDecorator<S, T> {
    // ONLY handles timeout protection
    TimeoutResult integrate_with_timeout(state& s, T dt, T end);
};
```

#### Parallel Facility
```cpp
class ParallelDecorator : public IntegratorDecorator<S, T> {
    // ONLY handles parallel execution
    void integrate_batch(StateRange&& states, T dt, T end);
    auto integrate_monte_carlo(size_t num_sims, Generator gen, Processor proc);
};
```

#### Output Facility
```cpp
class OutputDecorator : public IntegratorDecorator<S, T> {
    // ONLY handles output streaming/buffering
    // Supports ONLINE, OFFLINE, HYBRID modes
};
```

#### Signal Facility
```cpp
class SignalDecorator : public IntegratorDecorator<S, T> {
    // ONLY handles signal processing
    void register_signal_handler(std::function<void(S&, T)> handler);
};
```

### 3. Composition Builder (Low Coupling)

The `IntegratorBuilder` allows flexible composition:

```cpp
template<system_state S, can_be_time T = double>
class IntegratorBuilder {
public:
    IntegratorBuilder& with_timeout(TimeoutConfig config = {});
    IntegratorBuilder& with_parallel(ParallelConfig config = {});
    IntegratorBuilder& with_output(OutputConfig config = {});
    IntegratorBuilder& with_signals(SignalConfig config = {});
    
    std::unique_ptr<AbstractIntegrator<S, T>> build();
};
```

## Usage Examples

### Simple Composition

```cpp
// Just timeout protection
auto timeout_integrator = make_builder(base_integrator)
    .with_timeout(TimeoutConfig{.timeout_duration = std::chrono::seconds{30}})
    .build();

// Timeout + signals
auto realtime_integrator = make_builder(base_integrator)
    .with_timeout()
    .with_signals()
    .build();
```

### Complex Composition

```cpp
// All facilities combined
auto ultimate_integrator = make_builder(base_integrator)
    .with_timeout(TimeoutConfig{
        .timeout_duration = std::chrono::minutes{5},
        .enable_progress_callback = true
    })
    .with_parallel(ParallelConfig{.max_threads = 8})
    .with_signals(SignalConfig{.enable_real_time_processing = true})
    .with_output(OutputConfig{
        .mode = OutputMode::HYBRID,
        .output_interval = std::chrono::milliseconds{100}
    }, [](const auto& state, double t, size_t step) {
        std::cout << "t=" << t << ", state=" << state[0] << std::endl;
    })
    .build();
```

### Order Independence

The decorator pattern ensures composition order doesn't matter:

```cpp
// These are equivalent:
auto integrator1 = make_builder(base)
    .with_timeout().with_parallel().with_output().build();

auto integrator2 = make_builder(base)
    .with_output().with_timeout().with_parallel().build();

auto integrator3 = make_builder(base)
    .with_parallel().with_output().with_timeout().build();
```

## Benefits

### 1. Solves Combinatorial Explosion

- **Traditional approach**: 2^N classes for N facilities
- **Composable approach**: N classes for N facilities
- **Example**: 5 facilities = 32 combinations with only 5 classes

### 2. Easy Extension

Adding new facilities requires no modification of existing code:

```cpp
// Add new facility
class NetworkDecorator : public IntegratorDecorator<S, T> {
    // Implementation for distributed integration
};

// Extend builder
IntegratorBuilder& with_network(NetworkConfig config = {}) {
    integrator_ = std::make_unique<NetworkDecorator<S, T>>(
        std::move(integrator_), std::move(config));
    return *this;
}

// Immediately works with all existing facilities
auto distributed_integrator = make_builder(base)
    .with_timeout()
    .with_parallel()
    .with_network()  // New facility
    .with_output()
    .build();
```

### 3. Minimal Performance Overhead

- Each decorator adds minimal indirection
- Only pay for what you use
- Virtual function calls are optimizable by compilers

### 4. Type Safety

- All decorators maintain the same interface
- Compile-time type checking
- No runtime type confusion

## Real-World Scenarios

### Research Computing
```cpp
auto research_integrator = make_builder(base)
    .with_timeout(TimeoutConfig{.timeout_duration = std::chrono::hours{24}})
    .with_parallel(ParallelConfig{.max_threads = 16})
    .with_output(OutputConfig{.mode = OutputMode::OFFLINE})
    .build();
```

### Real-time Control
```cpp
auto control_integrator = make_builder(base)
    .with_timeout(TimeoutConfig{.timeout_duration = std::chrono::milliseconds{10}})
    .with_signals()
    .build();
```

### Production Server
```cpp
auto server_integrator = make_builder(base)
    .with_timeout(TimeoutConfig{
        .timeout_duration = std::chrono::seconds{30},
        .throw_on_timeout = false  // Don't crash server
    })
    .with_output(OutputConfig{.mode = OutputMode::HYBRID})
    .build();
```

### Interactive Application
```cpp
auto interactive_integrator = make_builder(base)
    .with_timeout(TimeoutConfig{
        .enable_progress_callback = true,
        .progress_callback = [](double current, double end, auto elapsed) {
            update_progress_bar(current / end);
            return !user_cancelled();
        }
    })
    .with_signals()
    .with_output()
    .build();
```

## Future Facilities

The architecture easily supports future additions:

- `CompressionDecorator`: State compression for memory efficiency
- `EncryptionDecorator`: Secure integration for sensitive data
- `NetworkDecorator`: Distributed integration across machines
- `GPUDecorator`: GPU acceleration for compute-intensive tasks
- `CachingDecorator`: Result caching for repeated computations
- `ProfilingDecorator`: Performance analysis and optimization
- `CheckpointDecorator`: Save/restore integration state
- `InterprocessDecorator`: IPC communication between processes

Each new facility automatically works with all existing ones without any code modification.

## Implementation Details

### Memory Management
- Uses `std::unique_ptr` for automatic memory management
- No manual memory management required
- Exception-safe construction/destruction

### Thread Safety
- Individual decorators handle their own thread safety
- Composition is thread-safe by design
- No shared mutable state between decorators

### Error Handling
- Each decorator can define its own error handling strategy
- Errors propagate naturally through the decorator chain
- Timeout errors can be configured to throw or return status

## Comparison with Alternatives

### vs. Template Mixins
- **Decorators**: Runtime composition, clear interfaces
- **Mixins**: Compile-time composition, potential diamond problem

### vs. Policy-Based Design
- **Decorators**: Easier to understand and debug
- **Policies**: More compile-time optimization potential

### vs. Inheritance Hierarchies
- **Decorators**: No multiple inheritance issues
- **Inheritance**: Potential for deep hierarchies and complexity

The decorator pattern strikes the best balance of flexibility, maintainability, and performance for this use case.

## Conclusion

The composable architecture successfully solves the combinatorial explosion problem while maintaining clean, maintainable code. It embodies the principles of:

- **Single Responsibility**: Each decorator has one job
- **Open/Closed**: Open for extension, closed for modification  
- **Composition over Inheritance**: Flexible runtime composition
- **Don't Repeat Yourself**: No duplicate combination classes

This design allows the DiffEq library to grow indefinitely in capabilities while keeping the codebase manageable and the user interface intuitive.