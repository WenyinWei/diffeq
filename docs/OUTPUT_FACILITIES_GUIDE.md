# Enhanced Output Facilities Guide

This guide covers the comprehensive output facilities added to the DiffEq library, providing dense output, interprocess communication, event-driven feedback, and SDE synchronization capabilities.

## Overview

The enhanced output facilities extend the existing composable architecture with powerful new capabilities:

- **Dense Output & Interpolation**: Query system state at arbitrary time points
- **Interprocess Communication**: Real-time data exchange between processes
- **Event-Driven Feedback**: Robotics control and sensor integration
- **SDE Synchronization**: Coordinated noise processes for stochastic systems

All facilities follow the same composable decorator pattern and can be combined in any order.

## Dense Output & Interpolation

### Basic Usage

```cpp
#include <core/composable_integration.hpp>

// Create integrator with dense output
auto dense_integrator = make_builder(std::move(base_integrator))
    .with_interpolation(InterpolationConfig{
        .method = InterpolationMethod::CUBIC_SPLINE,
        .max_history_size = 1000,
        .enable_compression = true
    })
    .build();

// Access interpolation capabilities
auto* interp = dynamic_cast<InterpolationDecorator<State, Time>*>(dense_integrator.get());

// Integrate normally
std::vector<double> state = {1.0, 0.0};
dense_integrator->integrate(state, 0.01, 2.0);

// Query state at arbitrary time
auto interpolated_state = interp->interpolate_at(1.5);

// Get dense output over interval
auto [times, states] = interp->get_dense_output(0.0, 2.0, 100);
```

### Interpolation Methods

- **LINEAR**: Fast linear interpolation
- **CUBIC_SPLINE**: Smooth cubic spline interpolation
- **HERMITE**: Hermite polynomial interpolation
- **AKIMA**: Akima spline (avoids oscillations)

### Configuration Options

```cpp
InterpolationConfig config;
config.method = InterpolationMethod::CUBIC_SPLINE;
config.max_history_size = 10000;           // Maximum stored points
config.enable_compression = true;           // Compress redundant points
config.compression_tolerance = 1e-8;        // Compression error tolerance
config.allow_extrapolation = false;         // Allow queries outside bounds
config.extrapolation_warning_threshold = 0.1; // Warn for extrapolation
```

## Interprocess Communication

### Shared Memory Communication

```cpp
// Producer process
InterprocessConfig producer_config;
producer_config.method = IPCMethod::SHARED_MEMORY;
producer_config.direction = IPCDirection::PRODUCER;
producer_config.channel_name = "simulation_data";
producer_config.buffer_size = 1024 * 1024;  // 1MB

auto producer = make_builder(std::move(integrator))
    .with_interprocess(producer_config)
    .build();

// Consumer process
InterprocessConfig consumer_config;
consumer_config.method = IPCMethod::SHARED_MEMORY;
consumer_config.direction = IPCDirection::CONSUMER;
consumer_config.channel_name = "simulation_data";
consumer_config.sync_mode = IPCSyncMode::BLOCKING;

auto consumer = make_builder(std::move(integrator))
    .with_interprocess(consumer_config)
    .build();
```

### Named Pipes Communication

```cpp
InterprocessConfig config;
config.method = IPCMethod::NAMED_PIPES;
config.direction = IPCDirection::BIDIRECTIONAL;
config.channel_name = "control_channel";
config.enable_acknowledgments = true;
config.max_retries = 3;

auto ipc_integrator = make_builder(std::move(integrator))
    .with_interprocess(config)
    .build();
```

### IPC Methods

- **SHARED_MEMORY**: Fastest, same-machine communication
- **NAMED_PIPES**: Cross-platform, moderate speed
- **MEMORY_MAPPED_FILE**: Persistent, file-based
- **TCP_SOCKET**: Network-capable (planned)
- **UDP_SOCKET**: Low-latency network (planned)

## Event-Driven Feedback

### Robotics Control Example

```cpp
// Configure for real-time control
EventConfig control_config;
control_config.processing_mode = EventProcessingMode::IMMEDIATE;
control_config.enable_control_loop = true;
control_config.control_loop_period = std::chrono::microseconds{1000}; // 1kHz
control_config.sensor_timeout = std::chrono::microseconds{2000};      // 2ms
control_config.enable_sensor_validation = true;

auto control_system = make_builder(std::move(robot_integrator))
    .with_events(control_config)
    .build();

auto* events = dynamic_cast<EventDecorator<State, Time>*>(control_system.get());

// Set up safety limits
events->set_threshold_event(0, 1.5, true, [](auto& state, auto time) {
    std::cout << "Joint limit exceeded! Emergency stop.\n";
    state[1] = 0.0;  // Stop motion
});

// Submit sensor data
events->submit_sensor_data("position_sensor", {1.2, 0.8}, 0.98);

// Submit control feedback
std::vector<double> target = {1.0, 0.5};
events->submit_control_feedback("pid_controller", target, current_state);
```

### Event Types

- **TIME_BASED**: Periodic time-triggered events
- **STATE_BASED**: Condition-based state events
- **SENSOR_DATA**: Sensor input events
- **CONTROL_FEEDBACK**: Control loop feedback
- **THRESHOLD_CROSSING**: Value crossing detection
- **CUSTOM**: User-defined events

### Event Priorities

- **LOW**: Background tasks
- **NORMAL**: Standard processing
- **HIGH**: Important events
- **CRITICAL**: Safety-critical events
- **EMERGENCY**: Immediate response required

## SDE Synchronization

### Coordinated Noise Processes

```cpp
#include <core/composable/sde_synchronization.hpp>

// Configure SDE synchronization
SDESyncConfig sync_config;
sync_config.sync_mode = SDESyncMode::BUFFERED;
sync_config.noise_type = NoiseProcessType::WIENER;
sync_config.noise_dimensions = 2;
sync_config.noise_intensity = 0.5;
sync_config.max_noise_delay = std::chrono::microseconds{1000};

// Create synchronized SDE pair
auto [producer, consumer] = SDESynchronizer<State, Time>::create_synchronized_pair(
    std::move(noise_generator),
    std::move(sde_integrator),
    ipc_config,
    sync_config
);
```

### Noise Process Types

- **WIENER**: Standard Brownian motion
- **COLORED_NOISE**: Correlated noise processes
- **JUMP_PROCESS**: Jump diffusion processes
- **LEVY_PROCESS**: Lévy processes
- **CUSTOM**: User-defined noise

### Synchronization Modes

- **IMMEDIATE**: Blocking until noise available
- **BUFFERED**: Buffer noise for smooth delivery
- **INTERPOLATED**: Interpolate between samples
- **GENERATED**: Local generation with sync seed

## Simultaneous Multiple Outputs

### Comprehensive Example

```cpp
// Single integrator with ALL output facilities
auto ultimate_integrator = make_builder(std::move(base_integrator))
    // Dense output for debugging
    .with_interpolation(InterpolationConfig{
        .method = InterpolationMethod::CUBIC_SPLINE,
        .max_history_size = 1000
    })
    // Real-time monitoring
    .with_output(OutputConfig{
        .mode = OutputMode::ONLINE,
        .output_interval = std::chrono::microseconds{100000}
    }, [](const auto& state, auto t, auto step) {
        std::cout << "Monitor: t=" << t << ", state=" << state[0] << "\n";
    })
    // Event-driven safety
    .with_events(EventConfig{
        .processing_mode = EventProcessingMode::IMMEDIATE,
        .max_event_processing_time = std::chrono::microseconds{500}
    })
    // IPC communication
    .with_interprocess(InterprocessConfig{
        .method = IPCMethod::SHARED_MEMORY,
        .direction = IPCDirection::PRODUCER,
        .channel_name = "realtime_data"
    })
    // Async processing
    .with_async(AsyncConfig{
        .thread_pool_size = 2
    })
    .build();
```

## Performance Considerations

### Dense Output
- Use LINEAR for fastest queries
- Enable compression for memory efficiency
- Limit history size for large simulations

### Interprocess Communication
- SHARED_MEMORY fastest for same-machine
- Named pipes for cross-platform
- Adjust buffer sizes for throughput

### Event Processing
- IMMEDIATE mode for real-time systems
- Limit processing time for hard real-time
- Use priority queues for mixed workloads

### SDE Synchronization
- BUFFERED mode for most applications
- IMMEDIATE for strict synchronization
- Monitor timeout rates for performance

## Real-World Applications

### Robotics Control
```cpp
auto robot_controller = make_builder(std::move(robot_dynamics))
    .with_events(EventConfig{.control_loop_period = std::chrono::microseconds{1000}})
    .with_interpolation(InterpolationConfig{.method = InterpolationMethod::CUBIC_SPLINE})
    .with_output(OutputConfig{.mode = OutputMode::ONLINE})
    .build();
```

### Financial Trading
```cpp
auto trading_system = make_builder(std::move(market_model))
    .with_events(EventConfig{.max_event_processing_time = std::chrono::microseconds{10}})
    .with_interprocess(InterprocessConfig{.method = IPCMethod::SHARED_MEMORY})
    .with_interpolation(InterpolationConfig{.method = InterpolationMethod::LINEAR})
    .build();
```

### Distributed SDE Simulation
```cpp
// Noise generator process
auto noise_producer = make_builder(std::move(wiener_generator))
    .with_interprocess(InterprocessConfig{.direction = IPCDirection::PRODUCER})
    .build();

// SDE integrator process
auto sde_consumer = configure_for_external_noise(
    std::move(sde_integrator), "wiener_channel");
```

### Scientific Computing
```cpp
auto research_integrator = make_builder(std::move(complex_system))
    .with_interpolation(InterpolationConfig{.max_history_size = 100000})
    .with_output(OutputConfig{.mode = OutputMode::OFFLINE})
    .with_timeout(TimeoutConfig{.timeout_duration = std::chrono::hours{24}})
    .build();
```

## Error Handling

### Graceful Degradation
- IPC failures fall back to local processing
- Event timeouts use default handlers
- Interpolation bounds checking with extrapolation warnings

### Debugging Support
- Comprehensive statistics for all facilities
- Event history tracking
- Performance metrics collection

### Validation
- Configuration validation at construction
- Runtime bounds checking
- Thread-safe error propagation

## Advanced Features

### Custom Interpolation
```cpp
class CustomInterpolator : public InterpolationDecorator<State, Time> {
    // Implement custom interpolation algorithm
};
```

### Custom IPC Channels
```cpp
class NetworkChannel : public IPCChannel<Time> {
    // Implement TCP/UDP networking
};
```

### Custom Event Triggers
```cpp
events->register_event_handler(EventTrigger::CUSTOM, 
    [](auto& state, auto time) {
        // Custom event logic
    });
```

## Best Practices

1. **Start Simple**: Begin with single decorators, combine as needed
2. **Profile Performance**: Monitor statistics for bottlenecks
3. **Validate Configurations**: Use config validation early
4. **Handle Errors Gracefully**: Plan for IPC failures and timeouts
5. **Use Appropriate Methods**: Match method to performance requirements
6. **Test Thoroughly**: Use provided unit tests and examples

## See Also

- [Composable Architecture Guide](COMPOSABLE_ARCHITECTURE.md)
- [Performance Optimization](PERFORMANCE_GUIDE.md)
- [Example Programs](../examples/)
- [API Reference](generated/) 