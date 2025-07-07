#pragma once

/**
 * @file composable_integration.hpp
 * @brief Composable integration architecture using decorator pattern
 * 
 * This header provides the main entry point for the composable integration
 * system. It includes all individual decorators and the builder interface.
 * 
 * The architecture solves the combinatorial explosion problem by using
 * high cohesion, low coupling principles:
 * 
 * - High Cohesion: Each decorator focuses on a single responsibility
 * - Low Coupling: Decorators can be combined in any order without dependencies
 * - Linear Scaling: N facilities require N classes, not 2^N classes
 * 
 * Usage Example:
 * ```cpp
 * auto integrator = make_builder(base_integrator)
 *     .with_timeout()
 *     .with_parallel()
 *     .with_async()
 *     .with_signals()
 *     .with_output()
 *     .build();
 * ```
 */

// Include all individual decorator components
#include "composable/integrator_decorator.hpp"
#include "composable/timeout_decorator.hpp"
#include "composable/parallel_decorator.hpp"
#include "composable/async_decorator.hpp"
#include "composable/output_decorator.hpp"
#include "composable/signal_decorator.hpp"
#include "composable/interpolation_decorator.hpp"  // Fixed template parameter issues
#include "composable/interprocess_decorator.hpp"   // Fixed template parameter issues  
// #include "composable/event_decorator.hpp"          // TODO: Fix remaining T template parameter references
// #include "composable/sde_synchronization.hpp"     // TODO: Fix remaining T template parameter references
// #include "composable/sde_multithreading.hpp"      // TODO: Fix remaining T template parameter references
#include "composable/integrator_builder.hpp"

namespace diffeq::core::composable {

/**
 * @brief Architecture summary
 * 
 * The composable integration architecture consists of:
 * 
 * 1. **Base Decorator (integrator_decorator.hpp)**
 *    - Foundation for all decorators
 *    - Implements delegation pattern
 *    - Provides common interface
 * 
 * 2. **Individual Facilities (one file each)**
 *    - TimeoutDecorator: Timeout protection
 *    - ParallelDecorator: Batch processing and Monte Carlo
 *    - AsyncDecorator: Asynchronous execution
 *    - OutputDecorator: Online/offline/hybrid output
 *    - SignalDecorator: Real-time signal processing
 *    - InterpolationDecorator: Dense output and interpolation
 *    - InterprocessDecorator: IPC communication
 *    - EventDecorator: Event-driven feedback and control
 * 
 * 3. **Composition Builder (integrator_builder.hpp)**
 *    - Fluent interface for combining decorators
 *    - Type-safe composition
 *    - Convenience functions for common patterns
 * 
 * Benefits:
 * - ✅ Solves combinatorial explosion (N classes vs 2^N)
 * - ✅ Order-independent composition
 * - ✅ Easy extensibility without modification
 * - ✅ Minimal performance overhead
 * - ✅ Type safety and compile-time checking
 */

} // namespace diffeq::core::composable