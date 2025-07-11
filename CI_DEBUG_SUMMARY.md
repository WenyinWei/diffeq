# CI Workflow Debug Summary

## Current Status

The CI workflows are failing due to template parameter issues in the composable integration decorators. The main issue is that many decorators were defined with two template parameters `<S, T>` but the base `AbstractIntegrator` and `IntegratorDecorator` classes only accept one parameter `<S>` (where the time type is derived from `typename S::value_type`).

## ✅ Issues Fixed

### 1. TimeoutDecorator
- **File**: `include/core/composable/timeout_decorator.hpp`
- **Fix**: Removed the `T` template parameter and updated all method signatures to use `typename IntegratorDecorator<S>::time_type`
- **Status**: ✅ COMPLETED

### 2. ParallelDecorator
- **File**: `include/core/composable/parallel_decorator.hpp`
- **Fix**: Removed the `T` template parameter and updated all method signatures
- **Status**: ✅ COMPLETED

### 3. OutputDecorator
- **File**: `include/core/composable/output_decorator.hpp`
- **Fix**: Removed the `T` template parameter and updated all method signatures
- **Status**: ✅ COMPLETED

### 4. SignalDecorator
- **File**: `include/core/composable/signal_decorator.hpp`
- **Fix**: Removed the `T` template parameter and updated all method signatures, including the SignalInfo struct
- **Status**: ✅ COMPLETED

## ⚠️ Issues Temporarily Disabled

These decorators have been temporarily commented out to allow basic compilation:

### 1. InterpolationDecorator
- **File**: `include/core/composable/interpolation_decorator.hpp`
- **Issue**: Still has many `T` template parameter references
- **Status**: 🚧 PARTIALLY FIXED - Need to complete the remaining T→time_type conversions

### 2. InterprocessDecorator
- **File**: `include/core/composable/interprocess_decorator.hpp`
- **Issue**: Still has `T` template parameter references and IPCChannel/IPCMessage template issues
- **Status**: 🚧 NEEDS FIXING

### 3. EventDecorator
- **File**: `include/core/composable/event_decorator.hpp`
- **Issue**: Template parameter issues and syntax errors in EventStats
- **Status**: 🚧 NEEDS FIXING

## 🔨 Files Modified for Temporary Fixes

### 1. Composable Integration Header
- **File**: `include/core/composable_integration.hpp`
- **Change**: Commented out problematic decorator includes
- **Reason**: Prevent compilation errors while fixing individual decorators

### 2. Integrator Builder
- **File**: `include/core/composable/integrator_builder.hpp`
- **Changes**: 
  - Commented out includes for problematic decorators
  - Commented out methods that reference disabled decorators
  - Commented out convenience functions for disabled decorators
- **Reason**: Prevent compilation errors in the builder system

## 🚨 Additional Issues Discovered

### 1. IntegratorBuilder Template Issues
- **File**: `include/core/composable/integrator_builder.hpp`
- **Issue**: Many `AbstractIntegrator<S, T>` references should be `AbstractIntegrator<S>`
- **Status**: 🚧 NEEDS FIXING

### 2. SDE Integrator Issues
- **Files**: All SDE integrator files in `include/integrators/sde/`
- **Issue**: References to missing `AbstractSDEIntegrator` class
- **Status**: 🚧 NEEDS INVESTIGATION

### 3. Async Integrator Issues
- **File**: `include/async/async_integrator.hpp`
- **Issue**: `AbstractIntegrator<S, T>` template parameter issues
- **Status**: 🚧 NEEDS FIXING

## 📋 Recommended Fix Plan

### Phase 1: Complete Template Parameter Fixes
1. Fix remaining `InterpolationDecorator` T references
2. Fix `InterprocessDecorator` template issues
3. Fix `EventDecorator` template and syntax issues
4. Fix `IntegratorBuilder` template parameter issues
5. Fix `AsyncIntegrator` template parameter issues

### Phase 2: Re-enable Disabled Components
1. Uncomment the fixed decorators in `composable_integration.hpp`
2. Uncomment the corresponding methods in `integrator_builder.hpp`
3. Test compilation of individual decorators

### Phase 3: SDE Integration Issues
1. Investigate missing `AbstractSDEIntegrator` base class
2. Fix SDE integrator inheritance issues
3. Update SDE integrator template parameters if needed

### Phase 4: Coverage and Package Issues
1. Fix coverage option in CI (`--enable_coverage=true` → proper coverage flags)
2. Fix package installation automation (auto-answer 'y' for package installs)
3. Add missing performance_benchmark target to xmake.lua

## 🔧 Template Parameter Pattern

The correct pattern for all decorators should be:

```cpp
// WRONG (old pattern):
template<system_state S, can_be_time T = double>
class MyDecorator : public IntegratorDecorator<S, T> {
    void method(typename IntegratorDecorator<S, T>::state_type& state, T dt, T end_time);
};

// CORRECT (new pattern):
template<system_state S>
class MyDecorator : public IntegratorDecorator<S> {
    void method(typename IntegratorDecorator<S>::state_type& state, 
                typename IntegratorDecorator<S>::time_type dt, 
                typename IntegratorDecorator<S>::time_type end_time);
};
```

## 🚀 Testing Strategy

1. **Simple Test**: Create minimal tests that use only basic integrators without decorators
2. **Incremental**: Enable decorators one by one as they're fixed
3. **Full Integration**: Test complete decorator composition once all are fixed

## ⏰ Time Estimate

- **Phase 1**: 2-3 hours (systematic template parameter fixes)
- **Phase 2**: 30 minutes (re-enabling components)
- **Phase 3**: 1-2 hours (SDE investigation and fixes)
- **Phase 4**: 1 hour (CI configuration fixes)

**Total Estimated Time**: 4-6 hours for complete resolution

## 💡 Prevention for Future

1. Add template parameter validation in CI
2. Create template usage guidelines in documentation
3. Consider using template aliases to reduce boilerplate
4. Add static_assert checks for template parameter consistency 