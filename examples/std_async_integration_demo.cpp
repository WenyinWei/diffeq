#define NOMINMAX
#include <diffeq.hpp>
#include <future>
#include <thread>
#include <chrono>
#include <iostream>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <functional>
#include <numeric>

#ifdef _WIN32
#include <windows.h>
#endif

/**
 * @brief 标准库异步积分示例
 * 
 * 这个示例展示了如何直接使用 C++ 标准库的异步设施进行微分方程积分，
 * 无需创建专门的管理器类，让代码更简单直接。
 * 
 * 核心理念：
 * - 直接使用 std::async 进行异步执行
 * - 使用 std::future 管理异步结果
 * - 专注于解决问题，而不是创建框架
 */

// 示例：参数化系统
struct SimpleSystem {
    double alpha, beta;
    
    SimpleSystem(double a, double b) : alpha(a), beta(b) {}
    
    void operator()(std::vector<double>& dx, const std::vector<double>& x, double t) const {
        dx[0] = alpha * x[0] - beta * x[0] * x[1];
        dx[1] = -beta * x[1] + alpha * x[0] * x[1];
    }
};

// 示例：数据分析器
class DataAnalyzer {
private:
    std::vector<std::pair<std::vector<double>, double>> data_;
    mutable std::mutex data_mutex_;

public:
    void analyze_result(const std::vector<double>& final_state, double final_time) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        // 计算简单的统计指标
        double magnitude = std::sqrt(final_state[0] * final_state[0] + final_state[1] * final_state[1]);
        double stability = std::abs(final_state[0] - final_state[1]);
        
        std::cout << "分析结果: 幅度=" << magnitude 
                  << ", 稳定性=" << stability 
                  << ", 时间=" << final_time << std::endl;
        
        data_.emplace_back(final_state, final_time);
    }
    
    const std::vector<std::pair<std::vector<double>, double>>& get_data() const {
        std::lock_guard<std::mutex> lock(data_mutex_);
        return data_;
    }
};

// 示例：轨迹保存器
class TrajectorySaver {
private:
    std::string prefix_;
    std::atomic<size_t> counter_{0};

public:
    explicit TrajectorySaver(std::string prefix = "traj_") : prefix_(std::move(prefix)) {}
    
    void save_trajectory(const std::vector<double>& final_state, double final_time) {
        auto count = ++counter_;
        std::string filename = prefix_ + std::to_string(count) + ".dat";
        
        // 模拟文件保存
        std::cout << "保存轨迹: " << filename 
                  << " (状态: [" << final_state[0] << ", " << final_state[1] 
                  << "], 时间: " << final_time << ")" << std::endl;
    }
};

// 示例：参数优化器
class ParameterOptimizer {
private:
    std::vector<double> best_params_;
    double best_objective_{std::numeric_limits<double>::max()};
    mutable std::mutex optimizer_mutex_;

public:
    void update_parameters(const std::vector<double>& params, double objective_value) {
        std::lock_guard<std::mutex> lock(optimizer_mutex_);
        
        if (objective_value < best_objective_) {
            best_objective_ = objective_value;
            best_params_ = params;
            
            std::cout << "发现更好的参数: [";
            for (size_t i = 0; i < params.size(); ++i) {
                std::cout << params[i];
                if (i < params.size() - 1) std::cout << ", ";
            }
            std::cout << "], 目标值: " << objective_value << std::endl;
        }
    }
    
    std::vector<double> get_best_parameters() const {
        std::lock_guard<std::mutex> lock(optimizer_mutex_);
        return best_params_;
    }
    
    double get_best_objective() const {
        std::lock_guard<std::mutex> lock(optimizer_mutex_);
        return best_objective_;
    }
};

int main() {
#ifdef _WIN32
    // 设置控制台编码为 UTF-8
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::cout << "=== 标准库异步积分示例 ===" << std::endl;
    std::cout << "展示如何直接使用标准库设施进行异步计算" << std::endl;
    
    // 创建后处理组件
    DataAnalyzer analyzer;
    TrajectorySaver saver("std_traj_");
    ParameterOptimizer optimizer;
    
    // 定义参数组合
    std::vector<std::pair<double, double>> parameters = {
        {0.5, 0.3}, {0.8, 0.2}, {0.3, 0.7}, {0.6, 0.4},
        {0.4, 0.6}, {0.7, 0.3}, {0.2, 0.8}, {0.9, 0.1}
    };
    
    std::cout << "\n启动 " << parameters.size() << " 个异步积分任务..." << std::endl;
    
    // 用于存储所有异步任务的 futures
    std::vector<std::future<void>> integration_futures;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 启动异步积分任务 - 直接使用 std::async
    for (size_t i = 0; i < parameters.size(); ++i) {
        const auto& [alpha, beta] = parameters[i];
        
        // 直接使用 std::async 启动异步任务
        auto future = std::async(std::launch::async,
            [&analyzer, &saver, &optimizer, alpha, beta, i]() {
                
                // 创建积分器和系统
                SimpleSystem system(alpha, beta);
                diffeq::RK4Integrator<std::vector<double>> integrator(
                    [system](double t, const std::vector<double>& x, std::vector<double>& dx) {
                        system(dx, x, t);
                    }
                );
                
                // 执行积分
                std::vector<double> state = {1.0, 0.5};
                integrator.integrate(state, 0.01, 10.0);
                
                std::cout << "任务 " << i << " 积分完成 (α=" << alpha << ", β=" << beta << ")" << std::endl;
                
                // 后处理任务 - 也使用异步执行
                std::vector<std::future<void>> post_futures;
                
                // 分析任务
                post_futures.push_back(std::async(std::launch::async, 
                    [&analyzer, state, time = integrator.current_time()]() {
                        analyzer.analyze_result(state, time);
                    }
                ));
                
                // 保存任务
                post_futures.push_back(std::async(std::launch::async, 
                    [&saver, state, time = integrator.current_time()]() {
                        saver.save_trajectory(state, time);
                    }
                ));
                
                // 优化任务
                double objective = std::abs(state[0] - state[1]);
                std::vector<double> params = {alpha, beta};
                
                post_futures.push_back(std::async(std::launch::async, 
                    [&optimizer, params, objective]() {
                        optimizer.update_parameters(params, objective);
                    }
                ));
                
                // 等待所有后处理任务完成
                for (auto& f : post_futures) {
                    f.wait();
                }
                
                std::cout << "任务 " << i << " 后处理完成" << std::endl;
            }
        );
        
        integration_futures.push_back(std::move(future));
    }
    
    // 在主线程执行一些其他任务（演示非阻塞）
    std::cout << "\n主线程可以继续执行其他任务..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "主线程的其他工作完成" << std::endl;
    
    // 等待所有积分任务完成
    std::cout << "\n等待所有异步任务完成..." << std::endl;
    for (auto& future : integration_futures) {
        future.wait();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // 显示结果
    std::cout << "\n=== 执行完成 ===" << std::endl;
    std::cout << "完成的任务: " << parameters.size() << std::endl;
    std::cout << "总耗时: " << duration.count() << "ms" << std::endl;
    std::cout << "平均每个任务: " << (duration.count() / parameters.size()) << "ms" << std::endl;
    
    // 显示优化结果
    auto best_params = optimizer.get_best_parameters();
    auto best_objective = optimizer.get_best_objective();
    
    if (!best_params.empty()) {
        std::cout << "\n=== 优化结果 ===" << std::endl;
        std::cout << "最佳参数: [";
        for (size_t i = 0; i < best_params.size(); ++i) {
            std::cout << best_params[i];
            if (i < best_params.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "最佳目标值: " << best_objective << std::endl;
    }
    
    // 显示分析结果
    const auto& analysis_data = analyzer.get_data();
    std::cout << "\n收集的分析数据点: " << analysis_data.size() << std::endl;
    
    // 示例：使用 std::packaged_task 进行更灵活的任务管理
    std::cout << "\n=== std::packaged_task 示例 ===" << std::endl;
    
    // 创建一个可以稍后执行的任务
    std::packaged_task<double(double, double)> task(
        [](double a, double b) -> double {
            // 执行一个简单的积分计算
            diffeq::RK4Integrator<std::vector<double>> integrator(
                [a, b](double t, const std::vector<double>& x, std::vector<double>& dx) {
                    dx[0] = a * x[0] - b * x[0] * x[1];
                    dx[1] = -b * x[1] + a * x[0] * x[1];
                }
            );
            
            std::vector<double> state = {2.0, 1.0};
            integrator.integrate(state, 0.01, 5.0);
            
            return std::sqrt(state[0] * state[0] + state[1] * state[1]);
        }
    );
    
    // 获取 future
    auto result_future = task.get_future();
    
    // 在另一个线程执行任务
    std::thread task_thread(std::move(task), 0.6, 0.4);
    
    // 等待结果
    double result = result_future.get();
    std::cout << "Packaged task 结果: " << result << std::endl;
    
    task_thread.join();
    
    return 0;
} 