cmake_minimum_required(VERSION 3.18)
project(diffeq VERSION 1.0 LANGUAGES CXX)

# 默认启用XTensor和Eigen（用户可通过-D关闭）
option(USE_XTENSOR "Enable XTensor backend" ON)
option(USE_EIGEN "Enable Eigen backend" ON)

# 条件检查依赖
include(CMakeFindDependencyMacro)
if(USE_XTENSOR)
    find_dependency(XTensor REQUIRED)
endif()
if(USE_EIGEN)
    find_dependency(Eigen3 3.3.9 REQUIRED)
endif()

# 核心库配置
add_library(diffeq_core INTERFACE)
target_include_directories(diffeq_core INTERFACE 
    $<$<BOOL:${USE_XTENSOR}>: ${XTENSOR_INCLUDE_DIRS}>
    $<$<BOOL:${USE_EIGEN}>: ${EIGEN3_INCLUDE_DIRS}>
)

# 插件系统
add_library(plugin_system INTERFACE)
target_link_libraries(plugin_system INTERFACE 
    $<$<BOOL:${USE_XTENSOR}>: xtensor_plugin>
    $<$<BOOL:${USE_EIGEN}>: eigen_plugin>
)