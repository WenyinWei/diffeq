set_languages("c++20")
-- project("diffeq")

-- 依赖管理
add_requires("eigen 3.4.0")
add_requires("gtest", {optional = true})

-- 核心库配置
target("diffeq_core")
    set_kind("shared")
    add_includedirs("include")
    add_links("diffeq_core")
    -- add_files("src/core/*.cpp") -- Commented out since there are no source files yet

-- 测试目标 - State概念测试
target("test_state_concept")
    set_kind("binary")
    add_includedirs("include")
    add_files("test/unit/test_state_concept.cpp")
    set_rundir("$(projectdir)")

-- 测试目标 - RK4积分器测试
target("test_rk4_integrator")
    set_kind("binary")
    add_includedirs("include")
    add_files("test/unit/test_rk4_integrator.cpp")
    set_rundir("$(projectdir)")

-- 测试目标 - 高级积分器测试
target("test_advanced_integrators")
    set_kind("binary")
    add_includedirs("include")
    add_files("test/unit/test_advanced_integrators.cpp")
    add_packages("gtest")
    set_rundir("$(projectdir)")

-- 示例目标 - State概念使用示例
target("state_concept_usage")
    set_kind("binary")
    add_includedirs("include")
    add_files("examples/state_concept_usage.cpp")
    set_rundir("$(projectdir)")

-- 示例目标 - RK4积分器使用示例
target("rk4_integrator_usage")
    set_kind("binary")
    add_includedirs("include")
    add_files("examples/rk4_integrator_usage.cpp")
    set_rundir("$(projectdir)")

-- 示例目标 - 高级积分器使用示例
target("advanced_integrators_usage")
    set_kind("binary")
    add_includedirs("include")
    add_files("examples/advanced_integrators_usage.cpp")
    set_rundir("$(projectdir)")

-- 示例目标 - 快速测试
target("quick_test")
    set_kind("binary")
    add_includedirs("include")
    add_files("examples/quick_test.cpp")
    set_rundir("$(projectdir)")

-- 示例目标 - DOP853特定测试
target("test_dop853_example")
    set_kind("binary")
    add_includedirs("include")
    add_files("examples/test_dop853.cpp")
    set_rundir("$(projectdir)")

-- 测试目标 - DOP853 综合测试
target("test_dop853")
    set_kind("binary")
    add_includedirs("include")
    add_files("test/unit/test_dop853.cpp")
    add_packages("gtest")
    set_rundir("$(projectdir)")

-- 测试目标 - SDE 积分器测试
target("test_sde_solvers")
    set_kind("binary")
    add_includedirs("include")
    add_files("test/integration/test_sde_solvers.cpp")
    add_packages("gtest")
    set_rundir("$(projectdir)")

-- 测试目标 - SDE 集成测试
target("test_sde_integration")
    set_kind("binary")
    add_includedirs("include")
    add_files("test/integration/test_sde_integration.cpp")
    add_packages("gtest")
    set_rundir("$(projectdir)")

-- 测试目标 - 现代化接口测试
target("test_modernized_interface")
    set_kind("binary")
    add_includedirs("include")
    add_files("test/integration/test_modernized_interface.cpp")
    add_packages("gtest")
    set_rundir("$(projectdir)")

-- 示例目标 - SDE 演示
target("sde_demo")
    set_kind("binary")
    add_includedirs("include")
    add_files("examples/sde_demo.cpp")
    set_rundir("$(projectdir)")

-- 自定义任务：运行所有测试
task("test")
    set_menu {
        usage = "xmake test",
        description = "Run all tests",
        options = {}
    }
    on_run(function ()
        import("core.base.task")
        
        -- 构建并运行测试
        print("Building and running State concept tests...")
        task.run("run", {}, "test_state_concept")
        
        print("\nBuilding and running RK4 integrator tests...")
        task.run("run", {}, "test_rk4_integrator")
        
        print("\nBuilding and running advanced integrators tests...")
        task.run("run", {}, "test_advanced_integrators")
        
        print("\nBuilding and running DOP853 comprehensive tests...")
        task.run("run", {}, "test_dop853")
        
        print("\nBuilding and running SDE solvers tests...")
        task.run("run", {}, "test_sde_solvers")
        
        print("\nBuilding and running SDE integration tests...")
        task.run("run", {}, "test_sde_integration")
        
        print("\nBuilding and running modernized interface tests...")
        task.run("run", {}, "test_modernized_interface")
    end)

-- 自定义任务：运行示例
task("example")
    set_menu {
        usage = "xmake example",
        description = "Run all usage examples",
        options = {}
    }
    on_run(function ()
        import("core.base.task")
        
        -- 构建并运行示例
        print("Building and running State concept example...")
        task.run("run", {}, "state_concept_usage")
        
        print("\nBuilding and running RK4 integrator example...")
        task.run("run", {}, "rk4_integrator_usage")
        
        print("\nBuilding and running advanced integrators example...")
        task.run("run", {}, "advanced_integrators_usage")
        
        print("\nBuilding and running SDE demo...")
        task.run("run", {}, "sde_demo")
    end)

-- 快速测试任务：只测试基本功能
task("quick-test")
    set_menu {
        usage = "xmake quick-test",
        description = "Run quick tests (concepts and RK4)",
        options = {}
    }
    on_run(function ()
        import("core.base.task")
        
        print("Building and running quick tests...")
        task.run("run", {}, "test_state_concept")
        task.run("run", {}, "test_rk4_integrator")
    end)

-- 演示任务：运行高级积分器演示
task("demo")
    set_menu {
        usage = "xmake demo",
        description = "Run advanced integrators demonstration",
        options = {}
    }
    on_run(function ()
        import("core.base.task")
        
        print("Building and running advanced integrators demonstration...")
        task.run("run", {}, "advanced_integrators_usage")
    end)

-- 快速DOP853测试任务：只测试DOP853
task("test-dop853")
    set_menu {
        usage = "xmake test-dop853",
        description = "Run DOP853 specific tests with timeout",
        options = {}
    }
    on_run(function ()
        import("core.base.task")
        
        print("Building and running DOP853 comprehensive tests...")
        task.run("run", {}, "test_dop853")
        
        print("\nRunning DOP853 example...")
        task.run("run", {}, "test_dop853_example")
    end)