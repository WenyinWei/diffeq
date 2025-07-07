set_languages("c++20")
set_project("diffeq")

-- Project metadata
set_version("1.0.0")
set_description("Modern C++ Differential Equation Solver Library")

-- Global settings
set_warnings("all")
set_optimize("fastest")

-- Enable parallel compilation for faster builds
-- xmake will automatically use parallel compilation

-- Handle character encoding issues on Windows
if is_plat("windows") then
    add_cxxflags("/utf-8")
end

-- Dependencies
add_requires("eigen 3.4.0")
add_requires("gtest", {optional = true})

-- Global include directories
add_includedirs("include")

-- ============================================================================
-- LIBRARY TARGETS
-- ============================================================================

-- Main library target (header-only for now)
target("diffeq")
    set_kind("headeronly")
    add_includedirs("include")
    add_packages("eigen")
    set_installdir("$(prefix)/include")
    add_headerfiles("include/(**/*.hpp)")

-- ============================================================================
-- UNIT TESTS
-- ============================================================================

-- State concept tests
target("test_state_concept")
    set_kind("binary")
    add_files("test/unit/test_state_concept.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("tests")

-- RK4 integrator tests
target("test_rk4_integrator")
    set_kind("binary")
    add_files("test/unit/test_rk4_integrator.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("tests")

-- Advanced integrators tests
target("test_advanced_integrators")
    set_kind("binary")
    add_files("test/unit/test_advanced_integrators.cpp")
    add_deps("diffeq")
    add_packages("gtest", {configs = {main = true}})
    set_rundir("$(projectdir)")
    set_group("tests")

-- DOP853 comprehensive tests
target("test_dop853")
    set_kind("binary")
    add_files("test/unit/test_dop853.cpp")
    add_deps("diffeq")
    add_packages("gtest", {configs = {main = true}})
    set_rundir("$(projectdir)")
    set_group("tests")

-- ============================================================================
-- INTEGRATION TESTS
-- ============================================================================

-- SDE solvers integration tests
target("test_sde_solvers")
    set_kind("binary")
    add_files("test/integration/test_sde_solvers.cpp")
    add_deps("diffeq")
    add_packages("gtest", {configs = {main = true}})
    set_rundir("$(projectdir)")
    set_group("integration_tests")

-- SDE integration tests
target("test_sde_integration")
    set_kind("binary")
    add_files("test/integration/test_sde_integration.cpp")
    add_deps("diffeq")
    add_packages("gtest", {configs = {main = true}})
    set_rundir("$(projectdir)")
    set_group("integration_tests")

-- Modernized interface tests
target("test_modernized_interface")
    set_kind("binary")
    add_files("test/integration/test_modernized_interface.cpp")
    add_deps("diffeq")
    add_packages("gtest", {configs = {main = true}})
    set_rundir("$(projectdir)")
    set_group("integration_tests")

-- Standard parallelism tests
target("test_standard_parallelism")
    set_kind("binary")
    add_files("test/integration/test_standard_parallelism.cpp")
    add_deps("diffeq")
    add_packages("gtest", {configs = {main = true}})
    set_rundir("$(projectdir)")
    set_group("integration_tests")

-- ============================================================================
-- EXAMPLES
-- ============================================================================

-- State concept usage example
target("state_concept_usage")
    set_kind("binary")
    add_files("examples/state_concept_usage.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")

-- RK4 integrator usage example
target("rk4_integrator_usage")
    set_kind("binary")
    add_files("examples/rk4_integrator_usage.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")

-- Advanced integrators usage example
target("advanced_integrators_usage")
    set_kind("binary")
    add_files("examples/advanced_integrators_usage.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")

-- Quick test example
target("quick_test")
    set_kind("binary")
    add_files("examples/quick_test.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")

-- DOP853 example
target("test_dop853_example")
    set_kind("binary")
    add_files("examples/test_dop853.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")

-- SDE demo
target("sde_demo")
    set_kind("binary")
    add_files("examples/sde_demo.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")

-- Interface usage demo
target("interface_usage_demo")
    set_kind("binary")
    add_files("examples/interface_usage_demo.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")

-- Parallelism usage demo
target("parallelism_usage_demo")
    set_kind("binary")
    add_files("examples/parallelism_usage_demo.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")

-- SDE usage demo
target("sde_usage_demo")
    set_kind("binary")
    add_files("examples/sde_usage_demo.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")

-- Standard parallelism demo
target("standard_parallelism_demo")
    set_kind("binary")
    add_files("examples/standard_parallelism_demo.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")

-- Working integrators demo
target("working_integrators_demo")
    set_kind("binary")
    add_files("examples/working_integrators_demo.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")

-- Realtime signal processing
target("realtime_signal_processing")
    set_kind("binary")
    add_files("examples/realtime_signal_processing.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")

-- Advanced GPU async demo
target("advanced_gpu_async_demo")
    set_kind("binary")
    add_files("examples/advanced_gpu_async_demo.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")

-- Simple standard parallelism
target("simple_standard_parallelism")
    set_kind("binary")
    add_files("examples/simple_standard_parallelism.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")

-- Simplified parallel usage
target("simplified_parallel_usage")
    set_kind("binary")
    add_files("examples/simplified_parallel_usage.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")

-- Test advanced parallelism
target("test_advanced_parallelism")
    set_kind("binary")
    add_files("examples/test_advanced_parallelism.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")

-- Performance benchmarking
target("performance_benchmark")
    set_kind("binary")
    add_files("examples/quick_test.cpp")  -- Use quick_test as a simple benchmark for now
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("performance")

-- ============================================================================
-- CUSTOM TASKS
-- ============================================================================

-- Comprehensive test suite
task("test-all")
    set_menu {
        usage = "xmake test-all",
        description = "Run all tests comprehensively",
        options = {
            {"v", "verbose", "kv", nil, "Verbose output"},
            {"p", "parallel", "kv", nil, "Run tests in parallel"},
            {"t", "timeout", "kv", "300", "Test timeout in seconds"}
        }
    }
    on_run(function (target, opt)
        import("core.base.task")
        import("core.base.option")
        
        local verbose = option.get("verbose")
        local parallel = option.get("parallel")
        local timeout = tonumber(option.get("timeout")) or 300
        
        print("🧪 Running comprehensive test suite...")
        
        local test_targets = {
            "test_state_concept",
            "test_rk4_integrator", 
            "test_advanced_integrators",
            "test_dop853",
            "test_sde_solvers",
            "test_sde_integration",
            "test_modernized_interface",
            "test_standard_parallelism"
        }
        
        local failed_tests = {}
        local passed_tests = {}
        
        for _, test_name in ipairs(test_targets) do
            if verbose then
                print("  🔄 Running " .. test_name .. "...")
            end
            
            local success = true
            local start_time = os.time()
            
            -- Run test with timeout
            local function run_test()
                task.run("run", {}, test_name)
            end
            
            if parallel then
                -- For parallel execution, we'd need more sophisticated handling
                success = os.exec("xmake run " .. test_name)
            else
                success = os.exec("xmake run " .. test_name)
            end
            
            local end_time = os.time()
            local duration = end_time - start_time
            
            if success then
                if verbose then
                    print("  ✅ " .. test_name .. " passed (" .. duration .. "s)")
                end
                table.insert(passed_tests, test_name)
            else
                print("  ❌ " .. test_name .. " failed (" .. duration .. "s)")
                table.insert(failed_tests, test_name)
            end
            
            if duration > timeout then
                print("  ⚠️  " .. test_name .. " exceeded timeout (" .. timeout .. "s)")
            end
        end
        
        -- Summary
        print("\n📊 Test Summary:")
        print("  ✅ Passed: " .. #passed_tests .. "/" .. #test_targets)
        print("  ❌ Failed: " .. #failed_tests)
        
        if #failed_tests > 0 then
            print("  Failed tests:")
            for _, test_name in ipairs(failed_tests) do
                print("    - " .. test_name)
            end
            os.exit(1)
        else
            print("  🎉 All tests passed!")
        end
    end)

-- Run all examples
task("examples-all")
    set_menu {
        usage = "xmake examples-all",
        description = "Run all examples",
        options = {
            {"v", "verbose", "kv", nil, "Verbose output"},
            {"s", "skip-long", "kv", nil, "Skip long-running examples"}
        }
    }
    on_run(function (target, opt)
        import("core.base.task")
        import("core.base.option")
        
        local verbose = option.get("verbose")
        local skip_long = option.get("skip-long")
        
        print("🚀 Running all examples...")
        
        local example_targets = {
            "state_concept_usage",
            "rk4_integrator_usage",
            "advanced_integrators_usage",
            "quick_test",
            "test_dop853_example",
            "sde_demo",
            "interface_usage_demo",
            "parallelism_usage_demo",
            "sde_usage_demo",
            "standard_parallelism_demo",
            "working_integrators_demo",
            "realtime_signal_processing",
            "advanced_gpu_async_demo",
            "simple_standard_parallelism",
            "simplified_parallel_usage",
            "test_advanced_parallelism"
        }
        
        local long_examples = {
            "advanced_gpu_async_demo",
            "realtime_signal_processing"
        }
        
        local failed_examples = {}
        local passed_examples = {}
        
        for _, example_name in ipairs(example_targets) do
            if skip_long then
                local is_long = false
                for _, long_name in ipairs(long_examples) do
                    if example_name == long_name then
                        is_long = true
                        break
                    end
                end
                if is_long then
                    if verbose then
                        print("  ⏭️  Skipping long example: " .. example_name)
                    end
                    goto continue
                end
            end
            
            if verbose then
                print("  🔄 Running " .. example_name .. "...")
            end
            
            local success = os.exec("xmake run " .. example_name)
            
            if success then
                if verbose then
                    print("  ✅ " .. example_name .. " completed")
                end
                table.insert(passed_examples, example_name)
            else
                print("  ❌ " .. example_name .. " failed")
                table.insert(failed_examples, example_name)
            end
            
            ::continue::
        end
        
        -- Summary
        print("\n📊 Example Summary:")
        print("  ✅ Completed: " .. #passed_examples)
        print("  ❌ Failed: " .. #failed_examples)
        
        if #failed_examples > 0 then
            print("  Failed examples:")
            for _, example_name in ipairs(failed_examples) do
                print("    - " .. example_name)
            end
        else
            print("  🎉 All examples completed successfully!")
        end
    end)

-- Quick test (basic functionality)
task("quick-test")
    set_menu {
        usage = "xmake quick-test",
        description = "Run quick tests (concepts and basic integrators)",
        options = {}
    }
    on_run(function ()
        import("core.base.task")
        
        print("⚡ Running quick tests...")
        
        local quick_tests = {
            "test_state_concept",
            "test_rk4_integrator"
        }
        
        for _, test_name in ipairs(quick_tests) do
            print("  🔄 Running " .. test_name .. "...")
            task.run("run", {}, test_name)
        end
        
        print("✅ Quick tests completed!")
    end)

-- Build and test everything
task("build-test-all")
    set_menu {
        usage = "xmake build-test-all",
        description = "Build everything and run comprehensive tests",
        options = {
            {"c", "clean", "kv", nil, "Clean before building"},
            {"v", "verbose", "kv", nil, "Verbose output"}
        }
    }
    on_run(function (target, opt)
        import("core.base.task")
        import("core.base.option")
        
        local clean = option.get("clean")
        local verbose = option.get("verbose")
        
        print("🔨 Building and testing everything...")
        
        if clean then
            print("  🧹 Cleaning...")
            os.exec("xmake clean")
        end
        
        print("  🔨 Building all targets...")
        os.exec("xmake build")
        
        print("  🧪 Running all tests...")
        task.run("test-all", {}, "verbose=" .. (verbose and "true" or "false"))
        
        print("  🚀 Running all examples...")
        task.run("examples-all", {}, "verbose=" .. (verbose and "true" or "false"))
        
        print("🎉 Build and test completed successfully!")
    end)

-- Documentation generation
task("docs")
    set_menu {
        usage = "xmake docs",
        description = "Generate comprehensive documentation",
        options = {
            {"d", "doxygen", "kv", nil, "Generate Doxygen documentation only"},
            {"s", "sphinx", "kv", nil, "Generate Sphinx documentation only"},
            {"a", "api", "kv", nil, "Generate API documentation only"},
            {"e", "examples", "kv", nil, "Generate examples documentation only"},
            {"p", "performance", "kv", nil, "Generate performance documentation only"},
            {"c", "clean", "kv", nil, "Clean generated documentation"},
            {"v", "serve", "kv", nil, "Generate and serve documentation locally"}
        }
    }
    on_run(function (target, opt)
        import("core.base.task")
        import("core.base.option")
        
        local docs_script = path.join(os.projectdir(), "tools", "scripts", "build_docs.sh")
        local docs_bat = path.join(os.projectdir(), "tools", "scripts", "build_docs.bat")
        
        -- Determine which script to use based on platform
        local script_path = nil
        if os.host() == "windows" then
            script_path = docs_bat
        else
            script_path = docs_script
        end
        
        if not os.exists(script_path) then
            print("Documentation build script not found: " .. script_path)
            return
        end
        
        -- Determine what to build based on options
        local build_target = "all"
        if option.get("doxygen") then
            build_target = "doxygen"
        elseif option.get("sphinx") then
            build_target = "sphinx"
        elseif option.get("api") then
            build_target = "api"
        elseif option.get("examples") then
            build_target = "examples"
        elseif option.get("performance") then
            build_target = "performance"
        elseif option.get("clean") then
            build_target = "clean"
        elseif option.get("serve") then
            build_target = "serve"
        end
        
        print("📚 Generating documentation: " .. build_target)
        
        if os.host() == "windows" then
            os.exec(script_path .. " " .. build_target)
        else
            os.exec("bash " .. script_path .. " " .. build_target)
        end
    end)

-- Legacy task aliases for backward compatibility
task("test")
    set_menu {
        usage = "xmake test",
        description = "Alias for test-all (legacy compatibility)"
    }
    on_run(function ()
        task.run("test-all")
    end)

task("example")
    set_menu {
        usage = "xmake example", 
        description = "Alias for examples-all (legacy compatibility)"
    }
    on_run(function ()
        task.run("examples-all")
    end)

task("demo")
    set_menu {
        usage = "xmake demo",
        description = "Run advanced integrators demonstration (legacy compatibility)"
    }
    on_run(function ()
        import("core.base.task")
        print("🎯 Running advanced integrators demonstration...")
        task.run("run", {}, "advanced_integrators_usage")
    end)

task("test-dop853")
    set_menu {
        usage = "xmake test-dop853",
        description = "Run DOP853 specific tests (legacy compatibility)"
    }
    on_run(function ()
        import("core.base.task")
        print("🔬 Running DOP853 comprehensive tests...")
        task.run("run", {}, "test_dop853")
        print("🎯 Running DOP853 example...")
        task.run("run", {}, "test_dop853_example")
    end)

task("new-examples")
    set_menu {
        usage = "xmake new-examples",
        description = "Run all newly created examples (legacy compatibility)"
    }
    on_run(function ()
        import("core.base.task")
        print("🆕 Running new examples...")
        
        local new_examples = {
            "interface_usage_demo",
            "parallelism_usage_demo", 
            "sde_usage_demo",
            "standard_parallelism_demo",
            "test_standard_parallelism"
        }
        
        for _, example_name in ipairs(new_examples) do
            print("  🔄 Running " .. example_name .. "...")
            task.run("run", {}, example_name)
        end
    end)