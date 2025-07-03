set_languages("c++20")  -- 设置C++标准
project(diffeq)  -- 项目名称
-- project("diffeq", {version = "0.1.0"})  -- 项目版本

add_requires("eigen 3.4.0")    -- 自动处理Eigen依

-- 核心库配置
target("diffeq_core")
    set_kind("shared")  -- 接口库
    add_includedirs("include/core")  -- 添加头文件目录
    add_links("diffeq_core")  -- 导出库名称

-- 正确option定义方式
option("use_xtensor", "Enable XTensor backend", true)
option("use_eigen", "Enable Eigen backend", true)


-- 接口库配置修正
target("diffeq_core")
    if has_config("use_eigen") then
        add_defines("ENABLE_EIGEN")
        print("Using Eigen backend")
    else
        add_defines("DISABLE_EIGEN")
    end