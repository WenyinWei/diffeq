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
add_requires("boost", {optional = true})

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

-- Integrator accuracy tests
target("test_integrator_accuracy")
    set_kind("binary")
    add_files("test/unit/test_integrator_accuracy.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("tests")

-- Simple accuracy test
target("test_simple_accuracy")
    set_kind("binary")
    add_files("test_simple_accuracy.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("tests")

-- ============================================================================
-- INTEGRATION TESTS
-- ============================================================================

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



-- Parallelism usage demo
target("parallelism_usage_demo")
    set_kind("binary")
    add_files("examples/parallelism_usage_demo.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")

-- Asio integration demo
target("asio_integration_demo")
    set_kind("binary")
    add_files("examples/asio_integration_demo.cpp")
    add_deps("diffeq")
    add_packages("boost")
    set_rundir("$(projectdir)")
    set_group("examples")

-- Advanced asio integration demo
target("advanced_asio_integration")
    set_kind("binary")
    add_files("examples/advanced_asio_integration.cpp")
    add_deps("diffeq")
    add_packages("boost")
    set_rundir("$(projectdir)")
    set_group("examples")

-- Standard library async integration demo
target("std_async_integration_demo")
    set_kind("binary")
    add_files("examples/std_async_integration_demo.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")

target("coroutine_integration_demo")
    set_kind("binary")
    add_files("examples/coroutine_integration_demo.cpp")
    add_deps("diffeq")
    set_rundir("$(projectdir)")
    set_group("examples")
    set_languages("c++20")  -- 确保启用 C++20 协程支持

-- SDE usage demo (temporarily disabled due to API issues)
-- target("sde_usage_demo")
--     set_kind("binary")
--     add_files("examples/sde_usage_demo.cpp")
--     add_deps("diffeq")
--     set_rundir("$(projectdir)")
--     set_group("examples")





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
        print("[1] Entered test-all task")
        import("core.base.task")
        import("core.base.option")
        print("[2] Imported task and option modules")
        -- Anti-recursion protection
        if os.getenv("XMAKE_TEST_ALL_RUNNING") then
            print("[3] Recursion detected, exiting test-all task")
            print("[ERROR] test-all task is already running! Possible infinite recursion detected.")
            print("   This can happen if test-all calls itself or is called from build-test-all.")
            print("   Please use 'xmake test-all' directly, not from other tasks.")
            os.exit(1)
        end
        print("[4] No recursion detected, setting env flag")
        -- Set environment variable to prevent recursion
        os.setenv("XMAKE_TEST_ALL_RUNNING", "1")
        
        local verbose = option.get("verbose")
        local parallel = option.get("parallel")
        local timeout = tonumber(option.get("timeout")) or 300
        print("[5] Parsed options: verbose=" .. tostring(verbose) .. ", parallel=" .. tostring(parallel) .. ", timeout=" .. tostring(timeout))
        print("[6] About to print test suite start message")
        print("[TEST] Running comprehensive test suite...")
        
        local test_targets = {
            "test_state_concept",
            "test_rk4_integrator", 
            "test_advanced_integrators",
            "test_integrator_accuracy",
            "test_standard_parallelism"
        }
        print("[7] Test targets table created")
        
        local failed_tests = {}
        local passed_tests = {}
        print("[8] Entering test loop")
        for i, test_name in ipairs(test_targets) do
            print("[9] Running test " .. i .. ": " .. test_name)
            if verbose then
                print("  [RUN] Running " .. test_name .. "...")
            end
            local success = true
            local start_time = os.time()
            print("[10] About to run test target: " .. test_name)
            -- Run test with timeout
            local function run_test()
                print("[11] (unused) run_test closure called for " .. test_name)
                task.run("run", {}, test_name)
            end
            if parallel then
                print("[12] Running in parallel mode")
                success = task.run("run", {}, test_name)
            else
                print("[13] Running in sequential mode")
                success = task.run("run", {}, test_name)
            end
            local end_time = os.time()
            local duration = end_time - start_time
            print("[14] Test " .. test_name .. " finished, duration: " .. duration .. "s, success: " .. tostring(success))
            -- In xmake, task.run returns true on success, false on failure
            -- But we need to check if the program actually ran successfully
            if success ~= false then  -- Changed from 'if success then' to handle nil/true cases
                if verbose then
                    print("  [PASS] " .. test_name .. " passed (" .. duration .. "s)")
                end
                table.insert(passed_tests, test_name)
            else
                print("  [FAIL] " .. test_name .. " failed (" .. duration .. "s)")
                table.insert(failed_tests, test_name)
            end
            if duration > timeout then
                print("  [WARN] " .. test_name .. " exceeded timeout (" .. timeout .. "s)")
            end
        end
        print("[15] Test loop finished, printing summary")
        -- Summary
        print("\n[SUMMARY] Test Summary:")
        print("  [PASS] Passed: " .. #passed_tests .. "/" .. #test_targets)
        print("  [FAIL] Failed: " .. #failed_tests)
        if #failed_tests > 0 then
            print("  Failed tests:")
            for _, test_name in ipairs(failed_tests) do
                print("    - " .. test_name)
            end
            print("[16] Exiting with failure")
            os.exit(1)
        else
            print("  [SUCCESS] All tests passed!")
            print("[17] Exiting with success")
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
        
        print("[RUN] Running all examples...")
        
        local example_targets = {
            "state_concept_usage",
            "rk4_integrator_usage",
            "advanced_integrators_usage",
            "parallelism_usage_demo",
            "asio_integration_demo",
            "advanced_asio_integration",
            "std_async_integration_demo",
            "coroutine_integration_demo"
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
                        print("  [SKIP] Skipping long example: " .. example_name)
                    end
                    goto continue
                end
            end
            
            if verbose then
                print("  [RUN] Running " .. example_name .. "...")
            end
            
            local success = task.run("run", {}, example_name)
            
            if success ~= false then  -- Changed from 'if success then' to handle nil/true cases
                if verbose then
                    print("  [PASS] " .. example_name .. " completed")
                end
                table.insert(passed_examples, example_name)
            else
                print("  [FAIL] " .. example_name .. " failed")
                table.insert(failed_examples, example_name)
            end
            
            ::continue::
        end
        
        -- Summary
        print("\n[SUMMARY] Example Summary:")
        print("  [PASS] Completed: " .. #passed_examples)
        print("  [FAIL] Failed: " .. #failed_examples)
        
        if #failed_examples > 0 then
            print("  Failed examples:")
            for _, example_name in ipairs(failed_examples) do
                print("    - " .. example_name)
            end
        else
            print("  [SUCCESS] All examples completed successfully!")
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
        
        print("[QUICK] Running quick tests...")
        
        local quick_tests = {
            "test_state_concept",
            "test_rk4_integrator"
        }
        
        for _, test_name in ipairs(quick_tests) do
            print("  [RUN] Running " .. test_name .. "...")
            task.run("run", {}, test_name)
        end
        
        print("[PASS] Quick tests completed!")
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
        
        print("[BUILD] Building and testing everything...")
        
        if clean then
            print("  🧹 Cleaning...")
            os.exec("xmake clean")
        end
        
        print("  [BUILD] Building all targets...")
        os.exec("xmake build")
        
        print("  [TEST] Running all tests...")
        -- Don't call test-all task to avoid recursion, run tests directly
        local test_targets = {
            "test_state_concept",
            "test_rk4_integrator", 
            "test_advanced_integrators",
            "test_integrator_accuracy",
            "test_standard_parallelism"
        }
        
        for _, test_name in ipairs(test_targets) do
            if verbose then
                print("    [RUN] Running " .. test_name .. "...")
            end
            task.run("run", {}, test_name)
            if verbose then
                print("    [PASS] " .. test_name .. " completed")
            end
        end
        
        print("  [RUN] Running all examples...")
        -- Don't call examples-all task to avoid potential recursion, run examples directly
        local example_targets = {
            "state_concept_usage",
            "rk4_integrator_usage",
            "advanced_integrators_usage",
            "parallelism_usage_demo"
        }
        
        for _, example_name in ipairs(example_targets) do
            if verbose then
                print("    [RUN] Running " .. example_name .. "...")
            end
            task.run("run", {}, example_name)
            if verbose then
                print("    [PASS] " .. example_name .. " completed")
            end
        end
        
        print("[SUCCESS] Build and test completed successfully!")
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
        
        print("[DOCS] Generating documentation: " .. build_target)
        
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
        -- Anti-recursion protection
        if os.getenv("XMAKE_TEST_ALL_RUNNING") then
            print("[ERROR] test-all task is already running! Possible infinite recursion detected.")
            print("   This can happen if test-all calls itself or is called from other tasks.")
            print("   Please use 'xmake test-all' directly, not from other tasks.")
            os.exit(1)
        end
        
        -- Set environment variable to prevent recursion
        os.setenv("XMAKE_TEST_ALL_RUNNING", "1")
        
        -- Run tests directly instead of calling test-all task
        local test_targets = {
            "test_state_concept",
            "test_rk4_integrator", 
            "test_advanced_integrators",
            "test_integrator_accuracy",
            "test_standard_parallelism"
        }
        
        print("[TEST] Running comprehensive test suite (via test alias)...")
        
        for _, test_name in ipairs(test_targets) do
            print("  [RUN] Running " .. test_name .. "...")
            task.run("run", {}, test_name)
            print("  [PASS] " .. test_name .. " completed")
        end
        
        print("[SUCCESS] All tests completed successfully!")
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
        print("[DEMO] Running advanced integrators demonstration...")
        task.run("run", {}, "advanced_integrators_usage")
    end)

task("test-dop853")
    set_menu {
        usage = "xmake test-dop853",
        description = "Run DOP853 specific tests (legacy compatibility)"
    }
    on_run(function ()
        import("core.base.task")
        print("[TEST] Running DOP853 comprehensive tests...")
        task.run("run", {}, "test_dop853")
        print("[DEMO] Running DOP853 example...")
        task.run("run", {}, "test_dop853_example")
    end)

task("new-examples")
    set_menu {
        usage = "xmake new-examples",
        description = "Run all newly created examples (legacy compatibility)"
    }
    on_run(function ()
        import("core.base.task")
        print("[NEW] Running new examples...")
        
        local new_examples = {
            "interface_usage_demo",
            "parallelism_usage_demo", 
            "sde_usage_demo",
            "standard_parallelism_demo",
            "test_standard_parallelism"
        }
        
        for _, example_name in ipairs(new_examples) do
            print("  [RUN] Running " .. example_name .. "...")
            task.run("run", {}, example_name)
        end
    end)

-- Test integrator accuracy
task("test-accuracy")
    set_menu {
        usage = "xmake test-accuracy",
        description = "Run comprehensive integrator accuracy tests",
        options = {
            {"v", "verbose", "kv", nil, "Verbose output"},
            {"q", "quick", "kv", nil, "Quick test with fewer iterations"}
        }
    }
    on_run(function (target, opt)
        import("core.base.task")
        import("core.base.option")
        
        local verbose = option.get("verbose")
        local quick = option.get("quick")
        
        print("[ACCURACY] Running integrator accuracy tests...")
        print("This may take several minutes to complete.")
        
        if quick then
            print("[QUICK] Running in quick mode with reduced iterations.")
        end
        
        -- Build the test if needed
        os.exec("xmake build test_integrator_accuracy")
        
        -- Run the accuracy test
        local success = task.run("run", {}, "test_integrator_accuracy")
        
        if success ~= false then
            print("[SUCCESS] Integrator accuracy tests completed successfully!")
        else
            print("[FAIL] Integrator accuracy tests failed!")
            os.exit(1)
        end
    end)