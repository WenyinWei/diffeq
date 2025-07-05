@echo off
REM DiffEq Library Documentation Build Script for Windows
REM This script generates comprehensive documentation for the diffeq C++ library

setlocal enabledelayedexpansion

REM Configuration
set "PROJECT_ROOT=%~dp0..\.."
set "DOCS_DIR=%PROJECT_ROOT%\docs"
set "GENERATED_DIR=%DOCS_DIR%\generated"
set "BUILD_DIR=%PROJECT_ROOT%\build"

REM Create necessary directories
if not exist "%GENERATED_DIR%" mkdir "%GENERATED_DIR%"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo === DiffEq Library Documentation Builder ===
echo Project root: %PROJECT_ROOT%
echo Docs directory: %DOCS_DIR%
echo Generated docs: %GENERATED_DIR%
echo.

REM Function to check if command exists
:command_exists
where %1 >nul 2>&1
if %errorlevel% equ 0 (
    exit /b 0
) else (
    exit /b 1
)

REM Function to install dependencies
:install_dependencies
echo Installing documentation dependencies...
echo Please install the following tools manually:
echo   - Doxygen: https://www.doxygen.nl/download.html
echo   - Graphviz: https://graphviz.org/download/
echo   - Python: https://www.python.org/downloads/
echo   - Sphinx: pip install sphinx sphinx-rtd-theme breathe exhale
echo.
echo After installation, run this script again.
pause
exit /b 1

REM Function to generate Doxygen documentation
:generate_doxygen
echo Generating Doxygen documentation...

call :command_exists doxygen
if %errorlevel% neq 0 (
    echo Doxygen not found. Installing...
    call :install_dependencies
)

cd /d "%PROJECT_ROOT%"

REM Run Doxygen
doxygen Doxyfile

if %errorlevel% equ 0 (
    echo ✓ Doxygen documentation generated successfully
    echo    HTML output: %GENERATED_DIR%\html\index.html
) else (
    echo ✗ Doxygen generation failed
    exit /b 1
)
exit /b 0

REM Function to generate Sphinx documentation
:generate_sphinx
echo Generating Sphinx documentation...

call :command_exists sphinx-build
if %errorlevel% neq 0 (
    echo Sphinx not found. Installing...
    call :install_dependencies
)

REM Create Sphinx configuration if it doesn't exist
if not exist "%DOCS_DIR%\sphinx\conf.py" (
    mkdir "%DOCS_DIR%\sphinx"
    cd /d "%DOCS_DIR%\sphinx"
    
    REM Initialize Sphinx project
    sphinx-quickstart -q -p "DiffEq" -a "DiffEq Team" -v "1.0.0" -r "1.0.0" -l "en" -n
    
    REM Configure for C++ and Breathe
    (
        echo # Configuration file for the Sphinx documentation builder
        echo.
        echo import os
        echo import sys
        echo sys.path.insert^(0, os.path.abspath^('.'^)^)
        echo.
        echo project = 'DiffEq'
        echo copyright = '2024, DiffEq Team'
        echo author = 'DiffEq Team'
        echo release = '1.0.0'
        echo.
        echo extensions = [
        echo     'sphinx.ext.autodoc',
        echo     'sphinx.ext.napoleon',
        echo     'sphinx.ext.viewcode',
        echo     'sphinx.ext.intersphinx',
        echo     'breathe',
        echo     'exhale'
        echo ]
        echo.
        echo templates_path = ['_templates']
        echo exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
        echo.
        echo html_theme = 'sphinx_rtd_theme'
        echo html_static_path = ['_static']
        echo.
        echo # Breathe configuration
        echo breathe_projects = {
        echo     "diffeq": "../generated/xml"
        echo }
        echo breathe_default_project = "diffeq"
        echo breathe_default_members = ^('members', 'undoc-members'^)
        echo.
        echo # Exhale configuration
        echo exhale_args = {
        echo     "containmentFolder":     "./api",
        echo     "rootFileName":          "library_root.rst",
        echo     "rootFileTitle":         "Library API",
        echo     "doxygenStripFromPath":  "..",
        echo     "createTreeView":        True,
        echo     "exhaleExecutesDoxygen": False,
        echo     "exhaleDoxygenStdin":    ""
        echo }
    ) > conf.py
    
    REM Create index.rst
    (
        echo DiffEq Documentation
        echo ===================
        echo.
        echo Welcome to the DiffEq library documentation!
        echo.
        echo .. toctree::
        echo    :maxdepth: 2
        echo    :caption: Contents:
        echo.
        echo    api/library_root
        echo.
        echo Indices and tables
        echo ==================
        echo.
        echo * :ref:`genindex`
        echo * :ref:`modindex`
        echo * :ref:`search`
    ) > index.rst
    
    echo ✓ Sphinx project initialized
)

REM Generate Sphinx documentation
cd /d "%DOCS_DIR%\sphinx"
make html

if %errorlevel% equ 0 (
    echo ✓ Sphinx documentation generated successfully
    echo    HTML output: %DOCS_DIR%\sphinx\_build\html\index.html
) else (
    echo ✗ Sphinx generation failed
    exit /b 1
)
exit /b 0

REM Function to generate API documentation
:generate_api_docs
echo Generating API documentation...

REM Create API documentation directory
mkdir "%DOCS_DIR%\api" 2>nul

REM Generate header documentation
cd /d "%PROJECT_ROOT%"

REM Create main API index
(
    echo # DiffEq API Documentation
    echo.
    echo This directory contains comprehensive API documentation for the DiffEq library.
    echo.
    echo ## Core Components
    echo.
    echo ### Integrators
    echo - [ODE Integrators](integrators/ode/README.md^) - Ordinary Differential Equation solvers
    echo - [SDE Integrators](integrators/sde/README.md^) - Stochastic Differential Equation solvers
    echo.
    echo ### Core Concepts
    echo - [State Management](core/state.md^) - State representation and management
    echo - [Concepts](core/concepts.md^) - C++20 concepts used throughout the library
    echo - [Event System](core/events.md^) - Event handling and signal processing
    echo.
    echo ### Interfaces
    echo - [Integration Interface](interfaces/integration_interface.md^) - Unified integration interface
    echo - [Plugin System](plugins/README.md^) - Plugin architecture and extensions
    echo.
    echo ## Quick Start
    echo.
    echo ```cpp
    echo #include ^<diffeq.hpp^>
    echo #include ^<vector^>
    echo.
    echo // Define your ODE
    echo auto my_ode = [](double t, const std::vector^<double^>^& y, std::vector^<double^>^& dydt^) {
    echo     dydt[0] = -y[0];  // Exponential decay
    echo };
    echo.
    echo // Create integrator and solve
    echo std::vector^<double^> y = {1.0};
    echo auto integrator = diffeq::make_rk45^<std::vector^<double^>^>(my_ode^);
    echo integrator.integrate^(y, 0.01, 1.0^);
    echo ```
    echo.
    echo ## Examples
    echo.
    echo See the [examples directory](../../examples/^) for complete working examples.
) > "%DOCS_DIR%\api\README.md"

echo ✓ API documentation structure created
exit /b 0

REM Function to generate examples documentation
:generate_examples_docs
echo Generating examples documentation...

mkdir "%DOCS_DIR%\examples" 2>nul

REM Create examples index
(
    echo # DiffEq Examples
    echo.
    echo This directory contains comprehensive examples demonstrating the usage of the DiffEq library.
    echo.
    echo ## Basic Examples
    echo.
    echo ### [Quick Test](quick_test.md^)
    echo Simple integration example to verify installation.
    echo.
    echo ### [State Concept Usage](state_concept_usage.md^)
    echo Demonstrates the state concept and different state types.
    echo.
    echo ### [RK4 Integrator Usage](rk4_integrator_usage.md^)
    echo Basic Runge-Kutta 4th order integrator usage.
    echo.
    echo ## Advanced Examples
    echo.
    echo ### [Advanced Integrators](advanced_integrators_usage.md^)
    echo Demonstrates adaptive step-size integrators and error control.
    echo.
    echo ### [Interface Usage Demo](interface_usage_demo.md^)
    echo Shows the unified integration interface for cross-domain applications.
    echo.
    echo ### [Parallelism Usage](parallelism_usage_demo.md^)
    echo Demonstrates parallel integration capabilities.
    echo.
    echo ## Domain-Specific Examples
    echo.
    echo ### [Finance Examples](finance_examples.md^)
    echo Portfolio optimization and financial modeling examples.
    echo.
    echo ### [Robotics Examples](robotics_examples.md^)
    echo Robot control and trajectory planning examples.
    echo.
    echo ### [Scientific Computing](scientific_examples.md^)
    echo Chemical reactions, physics simulations, and scientific computing examples.
    echo.
    echo ## SDE Examples
    echo.
    echo ### [SDE Demo](sde_demo.md^)
    echo Basic stochastic differential equation examples.
    echo.
    echo ### [Advanced SDE Usage](sde_usage_demo.md^)
    echo Advanced SDE methods and applications.
    echo.
    echo ## Running Examples
    echo.
    echo All examples can be built and run using xmake:
    echo.
    echo ```bash
    echo # Build all examples
    echo xmake
    echo.
    echo # Run specific example
    echo xmake run quick_test
    echo xmake run rk4_integrator_usage
    echo xmake run advanced_integrators_usage
    echo.
    echo # Run all examples
    echo xmake example
    echo ```
) > "%DOCS_DIR%\examples\README.md"

echo ✓ Examples documentation created
exit /b 0

REM Function to generate performance documentation
:generate_performance_docs
echo Generating performance documentation...

mkdir "%DOCS_DIR%\performance" 2>nul

(
    echo # Performance Guide
    echo.
    echo This guide covers performance considerations and optimization techniques for the DiffEq library.
    echo.
    echo ## Benchmarking
    echo.
    echo ### Running Benchmarks
    echo ```bash
    echo # Build and run benchmarks
    echo xmake run benchmark_ode
    echo xmake run benchmark_sde
    echo xmake run benchmark_parallel
    echo ```
    echo.
    echo ### Performance Metrics
    echo - Integration speed (steps/second^)
    echo - Memory usage
    echo - Accuracy vs. speed trade-offs
    echo - Parallel scaling efficiency
    echo.
    echo ## Optimization Tips
    echo.
    echo ### State Types
    echo - Use `std::array` for small, fixed-size states
    echo - Use `std::vector` for dynamic states
    echo - Consider custom state types for domain-specific optimizations
    echo.
    echo ### Integrator Selection
    echo - **RK4**: Fast, good for non-stiff problems
    echo - **RK45**: Adaptive, recommended default
    echo - **DOP853**: High accuracy, slower
    echo - **BDF**: For stiff systems
    echo.
    echo ### Memory Management
    echo - Reuse state vectors when possible
    echo - Use move semantics for large state transfers
    echo - Consider memory pools for high-frequency integration
    echo.
    echo ## Parallel Performance
    echo.
    echo ### Threading
    echo - Use `std::execution::par` for parallel integration
    echo - Balance thread count with problem size
    echo - Consider NUMA-aware allocation for large problems
    echo.
    echo ### GPU Acceleration
    echo - Use CUDA/OpenCL backends for large-scale problems
    echo - Batch multiple integrations for better GPU utilization
    echo - Profile memory transfers vs. computation
) > "%DOCS_DIR%\performance\README.md"

echo ✓ Performance documentation created
exit /b 0

REM Function to create documentation index
:create_docs_index
echo Creating documentation index...

(
    echo # DiffEq Documentation
    echo.
    echo Welcome to the comprehensive documentation for the DiffEq C++ library.
    echo.
    echo ## Quick Navigation
    echo.
    echo ### 📚 [API Reference](generated/html/index.html^)
    echo Complete API documentation with search and navigation.
    echo.
    echo ### 🚀 [Getting Started](getting_started.md^)
    echo Quick start guide and installation instructions.
    echo.
    echo ### 📖 [User Guide](user_guide.md^)
    echo Comprehensive user guide with examples and best practices.
    echo.
    echo ### 🔧 [API Documentation](api/README.md^)
    echo Detailed API reference organized by component.
    echo.
    echo ### 💡 [Examples](examples/README.md^)
    echo Working examples for all major features.
    echo.
    echo ### ⚡ [Performance Guide](performance/README.md^)
    echo Performance optimization and benchmarking.
    echo.
    echo ### 🔬 [Advanced Topics](advanced_topics.md^)
    echo Advanced usage patterns and customization.
    echo.
    echo ## Library Overview
    echo.
    echo DiffEq is a modern C++ library for solving ordinary and stochastic differential equations with:
    echo.
    echo - **High Performance**: Optimized integrators with minimal overhead
    echo - **Modern C++**: C++20/23 features with concepts and templates
    echo - **Cross-Domain**: Unified interface for finance, robotics, science
    echo - **Signal Processing**: Real-time event handling and async processing
    echo - **Extensible**: Plugin architecture for custom integrators and backends
    echo.
    echo ## Key Features
    echo.
    echo ### ODE Solvers
    echo - Fixed step: Euler, Improved Euler, RK4
    echo - Adaptive: RK23, RK45, DOP853
    echo - Stiff: BDF, Radau, LSODA
    echo.
    echo ### SDE Solvers
    echo - Basic: Euler-Maruyama, Milstein
    echo - Advanced: SRA, SRI, SOSRA, SOSRI
    echo - High-order: Strong order 1.5 methods
    echo.
    echo ### Signal Processing
    echo - Real-time event handling
    echo - Continuous and discrete influences
    echo - Async output streams
    echo - Cross-domain unified interface
    echo.
    echo ## Quick Example
    echo.
    echo ```cpp
    echo #include ^<diffeq.hpp^>
    echo #include ^<vector^>
    echo.
    echo // Define ODE: dy/dt = -y
    echo auto decay_ode = [](double t, const std::vector^<double^>^& y, std::vector^<double^>^& dydt^) {
    echo     dydt[0] = -y[0];
    echo };
    echo.
    echo // Solve with high-accuracy integrator
    echo std::vector^<double^> y = {1.0};
    echo auto integrator = diffeq::make_dop853^<std::vector^<double^>^>(decay_ode^);
    echo integrator.integrate^(y, 0.01, 1.0^);
    echo.
    echo std::cout ^<^< "Solution: " ^<^< y[0] ^<^< std::endl;
    echo ```
    echo.
    echo ## Building Documentation
    echo.
    echo ```bash
    echo # Generate all documentation
    echo tools\scripts\build_docs.bat
    echo.
    echo # Generate specific documentation
    echo tools\scripts\build_docs.bat doxygen
    echo tools\scripts\build_docs.bat sphinx
    echo tools\scripts\build_docs.bat api
    echo ```
    echo.
    echo ## Contributing
    echo.
    echo See [CONTRIBUTING.md](../CONTRIBUTING.md^) for guidelines on contributing to the documentation.
) > "%DOCS_DIR%\index.md"

echo ✓ Documentation index created
exit /b 0

REM Function to generate all documentation
:generate_all
echo Generating all documentation...

call :generate_doxygen
call :generate_sphinx
call :generate_api_docs
call :generate_examples_docs
call :generate_performance_docs
call :create_docs_index

echo ✓ All documentation generated successfully!
echo.
echo Documentation locations:
echo   • Doxygen HTML: %GENERATED_DIR%\html\index.html
echo   • Sphinx HTML: %DOCS_DIR%\sphinx\_build\html\index.html
echo   • API Docs: %DOCS_DIR%\api\README.md
echo   • Examples: %DOCS_DIR%\examples\README.md
echo   • Performance: %DOCS_DIR%\performance\README.md
echo   • Main Index: %DOCS_DIR%\index.md
exit /b 0

REM Function to clean generated documentation
:clean_docs
echo Cleaning generated documentation...
if exist "%GENERATED_DIR%" rmdir /s /q "%GENERATED_DIR%"
if exist "%DOCS_DIR%\sphinx\_build" rmdir /s /q "%DOCS_DIR%\sphinx\_build"
echo ✓ Documentation cleaned
exit /b 0

REM Function to serve documentation locally
:serve_docs
echo Serving documentation locally...

call :command_exists python
if %errorlevel% neq 0 (
    echo Python not found. Cannot serve documentation.
    exit /b 1
)

cd /d "%GENERATED_DIR%\html"
echo Starting HTTP server on http://localhost:8000
echo Press Ctrl+C to stop
python -m http.server 8000
exit /b 0

REM Main script logic
if "%1"=="" goto :generate_all
if "%1"=="doxygen" goto :generate_doxygen
if "%1"=="sphinx" goto :generate_sphinx
if "%1"=="api" goto :generate_api_docs
if "%1"=="examples" goto :generate_examples_docs
if "%1"=="performance" goto :generate_performance_docs
if "%1"=="all" goto :generate_all
if "%1"=="clean" goto :clean_docs
if "%1"=="serve" goto :serve_docs
if "%1"=="install" goto :install_dependencies

echo Usage: %0 [doxygen^|sphinx^|api^|examples^|performance^|all^|clean^|serve^|install]
echo.
echo Commands:
echo   doxygen    - Generate Doxygen documentation
echo   sphinx     - Generate Sphinx documentation
echo   api        - Generate API documentation
echo   examples   - Generate examples documentation
echo   performance- Generate performance documentation
echo   all        - Generate all documentation (default^)
echo   clean      - Clean generated documentation
echo   serve      - Generate and serve documentation locally
echo   install    - Install documentation dependencies
exit /b 1 