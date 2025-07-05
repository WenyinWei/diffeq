#!/usr/bin/env python3
"""
Simple test script for the DiffEq documentation system.
This script tests the basic functionality without requiring external dependencies.
"""

import os
import sys
from pathlib import Path

def test_documentation_structure():
    """Test that the documentation structure is properly set up."""
    print("Testing documentation structure...")
    
    # Check required files exist
    required_files = [
        "Doxyfile",
        "docs/README.md",
        "docs/templates/header_template.md",
        "tools/scripts/build_docs.sh",
        "tools/scripts/build_docs.bat",
        ".github/workflows/docs.yml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    else:
        print("✅ All required documentation files exist")
        return True

def test_doxygen_config():
    """Test that Doxygen configuration is valid."""
    print("Testing Doxygen configuration...")
    
    try:
        with open("Doxyfile", "r") as f:
            content = f.read()
        
        # Check for required Doxygen settings
        required_settings = [
            "PROJECT_NAME",
            "INPUT",
            "OUTPUT_DIRECTORY",
            "GENERATE_HTML"
        ]
        
        missing_settings = []
        for setting in required_settings:
            if setting not in content:
                missing_settings.append(setting)
        
        if missing_settings:
            print(f"❌ Missing Doxygen settings: {missing_settings}")
            return False
        else:
            print("✅ Doxygen configuration appears valid")
            return True
            
    except Exception as e:
        print(f"❌ Error reading Doxyfile: {e}")
        return False

def test_build_scripts():
    """Test that build scripts are properly formatted."""
    print("Testing build scripts...")
    
    # Test bash script
    try:
        with open("tools/scripts/build_docs.sh", "r") as f:
            bash_content = f.read()
        
        if "#!/bin/bash" in bash_content and "generate_doxygen" in bash_content:
            print("✅ Bash build script is properly formatted")
        else:
            print("❌ Bash build script appears incomplete")
            return False
            
    except Exception as e:
        print(f"❌ Error reading bash script: {e}")
        return False
    
    # Test batch script
    try:
        with open("tools/scripts/build_docs.bat", "r") as f:
            batch_content = f.read()
        
        if "generate_doxygen" in batch_content:
            print("✅ Batch build script is properly formatted")
        else:
            print("❌ Batch build script appears incomplete")
            return False
            
    except Exception as e:
        print(f"❌ Error reading batch script: {e}")
        return False
    
    return True

def test_github_actions():
    """Test that GitHub Actions workflow is properly configured."""
    print("Testing GitHub Actions workflow...")
    
    try:
        with open(".github/workflows/docs.yml", "r") as f:
            content = f.read()
        
        # Check for required workflow elements
        required_elements = [
            "name: Documentation",
            "on:",
            "jobs:",
            "build-docs:",
            "deploy-docs:"
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Missing workflow elements: {missing_elements}")
            return False
        else:
            print("✅ GitHub Actions workflow is properly configured")
            return True
            
    except Exception as e:
        print(f"❌ Error reading GitHub Actions workflow: {e}")
        return False

def test_xmake_integration():
    """Test that xmake integration is properly configured."""
    print("Testing xmake integration...")
    
    try:
        with open("xmake.lua", "r") as f:
            content = f.read()
        
        # Check for documentation tasks
        required_tasks = [
            'task("docs")',
            'task("docs-check")',
            'task("docs-deploy")'
        ]
        
        missing_tasks = []
        for task in required_tasks:
            if task not in content:
                missing_tasks.append(task)
        
        if missing_tasks:
            print(f"❌ Missing xmake tasks: {missing_tasks}")
            return False
        else:
            print("✅ xmake integration is properly configured")
            return True
            
    except Exception as e:
        print(f"❌ Error reading xmake.lua: {e}")
        return False

def create_sample_documentation():
    """Create sample documentation to test the system."""
    print("Creating sample documentation...")
    
    # Create docs directory structure
    os.makedirs("docs/api", exist_ok=True)
    os.makedirs("docs/examples", exist_ok=True)
    os.makedirs("docs/performance", exist_ok=True)
    
    # Create sample API documentation
    api_content = """# DiffEq API Documentation

This directory contains comprehensive API documentation for the DiffEq library.

## Core Components

### Integrators
- [ODE Integrators](integrators/ode/README.md) - Ordinary Differential Equation solvers
- [SDE Integrators](integrators/sde/README.md) - Stochastic Differential Equation solvers

### Core Concepts
- [State Management](core/state.md) - State representation and management
- [Concepts](core/concepts.md) - C++20 concepts used throughout the library
- [Event System](core/events.md) - Event handling and signal processing

## Quick Start

```cpp
#include <diffeq.hpp>
#include <vector>

// Define your ODE
auto my_ode = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];  // Exponential decay
};

// Create integrator and solve
std::vector<double> y = {1.0};
auto integrator = diffeq::make_rk45<std::vector<double>>(my_ode);
integrator.integrate(y, 0.01, 1.0);
```
"""
    
    with open("docs/api/README.md", "w") as f:
        f.write(api_content)
    
    # Create sample examples documentation
    examples_content = """# DiffEq Examples

This directory contains comprehensive examples demonstrating the usage of the DiffEq library.

## Basic Examples

### [Quick Test](quick_test.md)
Simple integration example to verify installation.

### [State Concept Usage](state_concept_usage.md)
Demonstrates the state concept and different state types.

## Running Examples

All examples can be built and run using xmake:

```bash
# Build all examples
xmake

# Run specific example
xmake run quick_test
xmake run rk4_integrator_usage

# Run all examples
xmake example
```
"""
    
    with open("docs/examples/README.md", "w") as f:
        f.write(examples_content)
    
    # Create sample performance documentation
    performance_content = """# Performance Guide

This guide covers performance considerations and optimization techniques for the DiffEq library.

## Benchmarking

### Running Benchmarks
```bash
# Build and run benchmarks
xmake run benchmark_ode
xmake run benchmark_sde
xmake run benchmark_parallel
```

## Optimization Tips

### State Types
- Use `std::array` for small, fixed-size states
- Use `std::vector` for dynamic states
- Consider custom state types for domain-specific optimizations

### Integrator Selection
- **RK4**: Fast, good for non-stiff problems
- **RK45**: Adaptive, recommended default
- **DOP853**: High accuracy, slower
- **BDF**: For stiff systems
"""
    
    with open("docs/performance/README.md", "w") as f:
        f.write(performance_content)
    
    print("✅ Sample documentation created")

def main():
    """Run all documentation tests."""
    print("=== DiffEq Documentation System Test ===\n")
    
    tests = [
        test_documentation_structure,
        test_doxygen_config,
        test_build_scripts,
        test_github_actions,
        test_xmake_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}\n")
    
    print(f"=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All documentation tests passed!")
        print("\nCreating sample documentation...")
        create_sample_documentation()
        print("\nDocumentation system is ready to use!")
        print("\nNext steps:")
        print("1. Install Doxygen: https://www.doxygen.nl/download.html")
        print("2. Install Sphinx: pip install sphinx sphinx-rtd-theme breathe exhale")
        print("3. Run: xmake docs")
        print("4. View documentation at: docs/generated/html/index.html")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 