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