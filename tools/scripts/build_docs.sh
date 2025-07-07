#!/bin/bash

# DiffEq Library Documentation Build Script
# This script generates comprehensive documentation for the diffeq C++ library

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCS_DIR="$PROJECT_ROOT/docs"
GENERATED_DIR="$DOCS_DIR/generated"
BUILD_DIR="$PROJECT_ROOT/build"

# Create necessary directories
mkdir -p "$GENERATED_DIR"
mkdir -p "$BUILD_DIR"

echo -e "${BLUE}=== DiffEq Library Documentation Builder ===${NC}"
echo "Project root: $PROJECT_ROOT"
echo "Docs directory: $DOCS_DIR"
echo "Generated docs: $GENERATED_DIR"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install dependencies (Ubuntu/Debian)
install_dependencies() {
    echo -e "${YELLOW}Installing documentation dependencies...${NC}"
    
    if command_exists apt-get; then
        sudo apt-get update
        sudo apt-get install -y doxygen graphviz plantuml python3-pip
    elif command_exists brew; then
        brew install doxygen graphviz plantuml
    elif command_exists pacman; then
        sudo pacman -S doxygen graphviz plantuml
    else
        echo -e "${RED}Package manager not supported. Please install doxygen, graphviz, and plantuml manually.${NC}"
        exit 1
    fi
    
    # Install Python dependencies
    pip3 install --user sphinx sphinx-rtd-theme breathe exhale
}

# Function to generate Doxygen documentation
generate_doxygen() {
    echo -e "${BLUE}Generating Doxygen documentation...${NC}"
    
    if ! command_exists doxygen; then
        echo -e "${RED}Doxygen not found. Installing...${NC}"
        install_dependencies
    fi
    
    cd "$PROJECT_ROOT"
    
    # Run Doxygen
    doxygen Doxyfile
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Doxygen documentation generated successfully${NC}"
        echo "   HTML output: $GENERATED_DIR/html/index.html"
    else
        echo -e "${RED}✗ Doxygen generation failed${NC}"
        exit 1
    fi
}

# Function to generate Sphinx documentation
generate_sphinx() {
    echo -e "${BLUE}Generating Sphinx documentation...${NC}"
    
    if ! command_exists sphinx-build; then
        echo -e "${RED}Sphinx not found. Installing...${NC}"
        install_dependencies
    fi
    
    # Create Sphinx configuration if it doesn't exist
    if [ ! -f "$DOCS_DIR/sphinx/conf.py" ]; then
        mkdir -p "$DOCS_DIR/sphinx"
        cd "$DOCS_DIR/sphinx"
        
        # Initialize Sphinx project
        sphinx-quickstart -q -p "DiffEq" -a "DiffEq Team" -v "1.0.0" -r "1.0.0" -l "en" --no-sep
        
        # Configure for C++ and Breathe
        cat > conf.py << 'EOF'
# Configuration file for the Sphinx documentation builder

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

project = 'DiffEq'
copyright = '2024, DiffEq Team'
author = 'DiffEq Team'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'breathe',
    'exhale'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Breathe configuration
breathe_projects = {
    "diffeq": "../generated/xml"
}
breathe_default_project = "diffeq"
breathe_default_members = ('members', 'undoc-members')

# Exhale configuration
exhale_args = {
    "containmentFolder":     "./api",
    "rootFileName":          "library_root.rst",
    "rootFileTitle":         "Library API",
    "doxygenStripFromPath":  "..",
    "createTreeView":        True,
    "exhaleExecutesDoxygen": False,
    "exhaleDoxygenStdin":    ""
}
EOF
        
        # Create index.rst
        cat > index.rst << 'EOF'
DiffEq Documentation
===================

Welcome to the DiffEq library documentation!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/library_root

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
EOF
        
        echo -e "${GREEN}✓ Sphinx project initialized${NC}"
    fi
    
    # Generate Sphinx documentation
    cd "$DOCS_DIR/sphinx"
    sphinx-build -b html . _build/html
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Sphinx documentation generated successfully${NC}"
        echo "   HTML output: $DOCS_DIR/sphinx/_build/html/index.html"
    else
        echo -e "${RED}✗ Sphinx generation failed${NC}"
        exit 1
    fi
}

# Function to generate API documentation
generate_api_docs() {
    echo -e "${BLUE}Generating API documentation...${NC}"
    
    # Create API documentation directory
    mkdir -p "$DOCS_DIR/api"
    
    # Generate header documentation
    cd "$PROJECT_ROOT"
    
    # Create main API index
    cat > "$DOCS_DIR/api/README.md" << 'EOF'
# DiffEq API Documentation

This directory contains comprehensive API documentation for the DiffEq library.

## Core Components

### Integrators
- [ODE Integrators](integrators/ode/README.md) - Ordinary Differential Equation solvers
- [SDE Integrators](integrators/sde/README.md) - Stochastic Differential Equation solvers

### Core Concepts
- [State Management](core/state.md) - State representation and management
- [Concepts](core/concepts.md) - C++20 concepts used throughout the library
- [Event System](core/events.md) - Event handling and signal processing

### Interfaces
- [Integration Interface](interfaces/integration_interface.md) - Unified integration interface
- [Plugin System](plugins/README.md) - Plugin architecture and extensions

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

## Examples

See the [examples directory](../../examples/) for complete working examples.
EOF
    
    echo -e "${GREEN}✓ API documentation structure created${NC}"
}

# Function to generate examples documentation
generate_examples_docs() {
    echo -e "${BLUE}Generating examples documentation...${NC}"
    
    mkdir -p "$DOCS_DIR/examples"
    
    # Create examples index
    cat > "$DOCS_DIR/examples/README.md" << 'EOF'
# DiffEq Examples

This directory contains comprehensive examples demonstrating the usage of the DiffEq library.

## Basic Examples

### [Quick Test](quick_test.md)
Simple integration example to verify installation.

### [State Concept Usage](state_concept_usage.md)
Demonstrates the state concept and different state types.

### [RK4 Integrator Usage](rk4_integrator_usage.md)
Basic Runge-Kutta 4th order integrator usage.

## Advanced Examples

### [Advanced Integrators](advanced_integrators_usage.md)
Demonstrates adaptive step-size integrators and error control.

### [Interface Usage Demo](interface_usage_demo.md)
Shows the unified integration interface for cross-domain applications.

### [Parallelism Usage](parallelism_usage_demo.md)
Demonstrates parallel integration capabilities.

## Domain-Specific Examples

### [Finance Examples](finance_examples.md)
Portfolio optimization and financial modeling examples.

### [Robotics Examples](robotics_examples.md)
Robot control and trajectory planning examples.

### [Scientific Computing](scientific_examples.md)
Chemical reactions, physics simulations, and scientific computing examples.

## SDE Examples

### [SDE Demo](sde_demo.md)
Basic stochastic differential equation examples.

### [Advanced SDE Usage](sde_usage_demo.md)
Advanced SDE methods and applications.

## Running Examples

All examples can be built and run using xmake:

```bash
# Build all examples
xmake

# Run specific example
xmake run quick_test
xmake run rk4_integrator_usage
xmake run advanced_integrators_usage

# Run all examples
xmake example
```
EOF
    
    echo -e "${GREEN}✓ Examples documentation created${NC}"
}

# Function to generate performance documentation
generate_performance_docs() {
    echo -e "${BLUE}Generating performance documentation...${NC}"
    
    mkdir -p "$DOCS_DIR/performance"
    
    cat > "$DOCS_DIR/performance/README.md" << 'EOF'
# Performance Guide

This guide covers performance considerations and optimization techniques for the DiffEq library.

## Benchmarking

### Running Benchmarks
```bash
# Build and run benchmarks
xmake run benchmark_ode
xmake run benchmark_sde
xmake run benchmark_parallel
```

### Performance Metrics
- Integration speed (steps/second)
- Memory usage
- Accuracy vs. speed trade-offs
- Parallel scaling efficiency

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

### Memory Management
- Reuse state vectors when possible
- Use move semantics for large state transfers
- Consider memory pools for high-frequency integration

## Parallel Performance

### Threading
- Use `std::execution::par` for parallel integration
- Balance thread count with problem size
- Consider NUMA-aware allocation for large problems

### GPU Acceleration
- Use CUDA/OpenCL backends for large-scale problems
- Batch multiple integrations for better GPU utilization
- Profile memory transfers vs. computation
EOF
    
    echo -e "${GREEN}✓ Performance documentation created${NC}"
}

# Function to create documentation index
create_docs_index() {
    echo -e "${BLUE}Creating documentation index...${NC}"
    
    cat > "$DOCS_DIR/index.md" << 'EOF'
# DiffEq Documentation

Welcome to the comprehensive documentation for the DiffEq C++ library.

## Quick Navigation

### 📚 [API Reference](generated/html/index.html)
Complete API documentation with search and navigation.

### 🚀 [Getting Started](getting_started.md)
Quick start guide and installation instructions.

### 📖 [User Guide](user_guide.md)
Comprehensive user guide with examples and best practices.

### 🔧 [API Documentation](api/README.md)
Detailed API reference organized by component.

### 💡 [Examples](examples/README.md)
Working examples for all major features.

### ⚡ [Performance Guide](performance/README.md)
Performance optimization and benchmarking.

### 🔬 [Advanced Topics](advanced_topics.md)
Advanced usage patterns and customization.

## Library Overview

DiffEq is a modern C++ library for solving ordinary and stochastic differential equations with:

- **High Performance**: Optimized integrators with minimal overhead
- **Modern C++**: C++20/23 features with concepts and templates
- **Cross-Domain**: Unified interface for finance, robotics, science
- **Signal Processing**: Real-time event handling and async processing
- **Extensible**: Plugin architecture for custom integrators and backends

## Key Features

### ODE Solvers
- Fixed step: Euler, Improved Euler, RK4
- Adaptive: RK23, RK45, DOP853
- Stiff: BDF, Radau, LSODA

### SDE Solvers
- Basic: Euler-Maruyama, Milstein
- Advanced: SRA, SRI, SOSRA, SOSRI
- High-order: Strong order 1.5 methods

### Signal Processing
- Real-time event handling
- Continuous and discrete influences
- Async output streams
- Cross-domain unified interface

## Quick Example

```cpp
#include <diffeq.hpp>
#include <vector>

// Define ODE: dy/dt = -y
auto decay_ode = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
};

// Solve with high-accuracy integrator
std::vector<double> y = {1.0};
auto integrator = diffeq::make_dop853<std::vector<double>>(decay_ode);
integrator.integrate(y, 0.01, 1.0);

std::cout << "Solution: " << y[0] << std::endl;
```

## Building Documentation

```bash
# Generate all documentation
./tools/scripts/build_docs.sh

# Generate specific documentation
./tools/scripts/build_docs.sh doxygen
./tools/scripts/build_docs.sh sphinx
./tools/scripts/build_docs.sh api
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing to the documentation.
EOF
    
    echo -e "${GREEN}✓ Documentation index created${NC}"
}

# Function to generate all documentation
generate_all() {
    echo -e "${BLUE}Generating all documentation...${NC}"
    
    generate_doxygen
    generate_sphinx
    generate_api_docs
    generate_examples_docs
    generate_performance_docs
    create_docs_index
    
    echo -e "${GREEN}✓ All documentation generated successfully!${NC}"
    echo ""
    echo -e "${BLUE}Documentation locations:${NC}"
    echo "  • Doxygen HTML: $GENERATED_DIR/html/index.html"
    echo "  • Sphinx HTML: $DOCS_DIR/sphinx/_build/html/index.html"
    echo "  • API Docs: $DOCS_DIR/api/README.md"
    echo "  • Examples: $DOCS_DIR/examples/README.md"
    echo "  • Performance: $DOCS_DIR/performance/README.md"
    echo "  • Main Index: $DOCS_DIR/index.md"
}

# Function to clean generated documentation
clean_docs() {
    echo -e "${YELLOW}Cleaning generated documentation...${NC}"
    rm -rf "$GENERATED_DIR"
    rm -rf "$DOCS_DIR/sphinx/_build"
    echo -e "${GREEN}✓ Documentation cleaned${NC}"
}

# Function to serve documentation locally
serve_docs() {
    echo -e "${BLUE}Serving documentation locally...${NC}"
    
    if command_exists python3; then
        cd "$GENERATED_DIR/html"
        echo -e "${GREEN}Starting HTTP server on http://localhost:8000${NC}"
        echo "Press Ctrl+C to stop"
        python3 -m http.server 8000
    else
        echo -e "${RED}Python3 not found. Cannot serve documentation.${NC}"
        exit 1
    fi
}

# Main script logic
case "${1:-all}" in
    "doxygen")
        generate_doxygen
        ;;
    "sphinx")
        generate_sphinx
        ;;
    "api")
        generate_api_docs
        ;;
    "examples")
        generate_examples_docs
        ;;
    "performance")
        generate_performance_docs
        ;;
    "all")
        generate_all
        ;;
    "clean")
        clean_docs
        ;;
    "serve")
        generate_all
        serve_docs
        ;;
    "install")
        install_dependencies
        ;;
    *)
        echo -e "${RED}Usage: $0 [doxygen|sphinx|api|examples|performance|all|clean|serve|install]${NC}"
        echo ""
        echo "Commands:"
        echo "  doxygen    - Generate Doxygen documentation"
        echo "  sphinx     - Generate Sphinx documentation"
        echo "  api        - Generate API documentation"
        echo "  examples   - Generate examples documentation"
        echo "  performance- Generate performance documentation"
        echo "  all        - Generate all documentation (default)"
        echo "  clean      - Clean generated documentation"
        echo "  serve      - Generate and serve documentation locally"
        echo "  install    - Install documentation dependencies"
        exit 1
        ;;
esac 