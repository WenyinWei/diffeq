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

-- 文档生成任务
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
        
        print("Generating documentation: " .. build_target)
        
        if os.host() == "windows" then
            os.exec(script_path .. " " .. build_target)
        else
            os.exec("bash " .. script_path .. " " .. build_target)
        end
    end)

-- 文档检查任务
task("docs-check")
    set_menu {
        usage = "xmake docs-check",
        description = "Check documentation quality and completeness",
        options = {}
    }
    on_run(function (target, opt)
        import("core.base.task")
        
        print("Checking documentation quality...")
        
        -- Check if required documentation files exist
        local required_files = {
            "docs/index.md",
            "docs/api/README.md",
            "docs/examples/README.md",
            "docs/performance/README.md",
            "Doxyfile"
        }
        
        for _, file in ipairs(required_files) do
            if not os.exists(file) then
                print("Missing required documentation file: " .. file)
                return
            end
        end
        
        -- Check if documentation can be generated
        print("Testing documentation generation...")
        task.run("docs", {}, "doxygen")
        
        print("Documentation check completed successfully!")
    end)

-- 文档部署任务
task("docs-deploy")
    set_menu {
        usage = "xmake docs-deploy",
        description = "Deploy documentation to GitHub Pages",
        options = {}
    }
    on_run(function (target, opt)
        import("core.base.task")
        
        print("Deploying documentation to GitHub Pages...")
        
        -- Generate all documentation first
        task.run("docs", {}, "all")
        
        -- Check if we're in a git repository
        if not os.exists(".git") then
            print("Not in a git repository. Cannot deploy.")
            return
        end
        
        -- Create gh-pages branch and deploy
        os.exec("git checkout -b gh-pages")
        os.exec("git add docs/generated/")
        os.exec("git commit -m 'Update documentation'")
        os.exec("git push origin gh-pages")
        os.exec("git checkout main")
        
        print("Documentation deployed to GitHub Pages!")
        print("Visit: https://your-username.github.io/diffeq/")
    end)