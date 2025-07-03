set_languages("c++20")
-- project("diffeq")

-- 依赖管理
add_requires("eigen 3.4.0")
-- add_requires("xtensor 0.24.0")

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

-- 示例目标 - State概念使用示例
target("state_concept_usage")
    set_kind("binary")
    add_includedirs("include")
    add_files("examples/state_concept_usage.cpp")
    set_rundir("$(projectdir)")

-- 自定义任务：运行所有测试
task("test")
    set_menu {
        usage = "xmake test",
        description = "Run all concept tests",
        options = {}
    }
    on_run(function ()
        import("core.base.task")
        
        -- 构建并运行测试
        print("Building and running State concept tests...")
        task.run("run", {}, "test_state_concept")
    end)

-- 自定义任务：运行示例
task("example")
    set_menu {
        usage = "xmake example",
        description = "Run State concept usage example",
        options = {}
    }
    on_run(function ()
        import("core.base.task")
        
        -- 构建并运行示例
        print("Building and running State concept example...")
        task.run("run", {}, "state_concept_usage")
    end)