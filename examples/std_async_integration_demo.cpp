#define NOMINMAX
#include <diffeq.hpp>
#include <future>
#include <thread>
#include <chrono>
#include <iostream>
#include <vector>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

#ifdef _WIN32
#include <windows.h>
#endif

/**
 * @brief 使用标准库异步设施的积分器管理器
 * 
 * 这个示例展示了如何利用 C++ 标准库的异步设施，
 * 避免重复发明轮子，专注于 ODE 计算完成后的任务编排。
 * 
 * 设计理念：
 * - 使用 std::async 和 std::future 进行异步执行
 * - 使用 std::thread 和 std::mutex 进行任务管理
 * - 专注于任务编排，而不是积分器内部的异步化
 */
template<typename State>
class StdAsyncIntegrationManager {
private:
    std::unique_ptr<diffeq::core::AbstractIntegrator<State>> integrator_;
    std::vector<std::future<void>> pending_tasks_;
    std::atomic<size_t> completed_tasks_{0};
    std::atomic<size_t> total_tasks_{0};
    
    // 任务队列管理
    std::queue<std::function<void()>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> shutdown_{false};
    std::thread worker_thread_;

public:
    /**
     * @brief 构造函数
     */
    explicit StdAsyncIntegrationManager(std::unique_ptr<diffeq::core::AbstractIntegrator<State>> integrator)
        : integrator_(std::move(integrator)) {
        
        if (!integrator_) {
            throw std::invalid_argument("Integrator cannot be null");
        }
        
        // 启动工作线程
        worker_thread_ = std::thread([this] { worker_loop(); });
    }

    /**
     * @brief 析构函数
     */
    ~StdAsyncIntegrationManager() {
        shutdown_.store(true);
        queue_cv_.notify_all();
        
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        
        // 等待所有任务完成
        wait_for_all_tasks();
    }

    /**
     * @brief 异步执行积分任务
     */
    template<typename PostTask>
    void integrate_async(State initial_state,
                        typename diffeq::core::AbstractIntegrator<State>::time_type dt,
                        typename diffeq::core::AbstractIntegrator<State>::time_type end_time,
                        PostTask&& post_task) {
        
        ++total_tasks_;
        
        // 使用 std::async 启动异步任务
        auto future = std::async(std::launch::async, 
            [this, initial_state = std::move(initial_state), dt, end_time, 
             task = std::forward<PostTask>(post_task)]() mutable {
            
            try {
                // 执行积分
                integrator_->set_time(0.0);
                integrator_->integrate(initial_state, dt, end_time);
                
                // 执行后处理任务
                task(initial_state, integrator_->current_time());
                
                ++completed_tasks_;
                
            } catch (const std::exception& e) {
                std::cerr << "Integration task failed: " << e.what() << std::endl;
            }
        });
        
        // 存储 future 以便后续等待
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending_tasks_.push_back(std::move(future));
    }

    /**
     * @brief 添加延迟任务到队列
     */
    template<typename Task>
    void queue_task(Task&& task) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push(std::forward<Task>(task));
        queue_cv_.notify_one();
    }

    /**
     * @brief 等待所有任务完成
     */
    void wait_for_all_tasks() {
        std::vector<std::future<void>> tasks;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            tasks = std::move(pending_tasks_);
        }
        
        for (auto& task : tasks) {
            if (task.valid()) {
                task.wait();
            }
        }
    }

    /**
     * @brief 获取进度信息
     */
    std::pair<size_t, size_t> get_progress() const {
        return {completed_tasks_.load(), total_tasks_.load()};
    }

    /**
     * @brief 重置统计
     */
    void reset_stats() {
        completed_tasks_.store(0);
        total_tasks_.store(0);
    }

private:
    /**
     * @brief 工作线程循环
     */
    void worker_loop() {
        while (!shutdown_.load()) {
            std::function<void()> task;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] { 
                    return !task_queue_.empty() || shutdown_.load(); 
                });
                
                if (shutdown_.load()) {
                    break;
                }
                
                if (!task_queue_.empty()) {
                    task = std::move(task_queue_.front());
                    task_queue_.pop();
                }
            }
            
            if (task) {
                try {
                    task();
                } catch (const std::exception& e) {
                    std::cerr << "Queued task failed: " << e.what() << std::endl;
                }
            }
        }
    }
};

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
    std::mutex data_mutex_;

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

    std::cout << "=== 标准库异步积分器集成示例 ===" << std::endl;
    std::cout << "展示如何利用标准库设施避免重复发明轮子" << std::endl;
    
    // 创建积分器
    auto system = [](double t, const std::vector<double>& x, std::vector<double>& dx) {
        // 简单的 Lotka-Volterra 系统
        dx[0] = 0.5 * x[0] - 0.3 * x[0] * x[1];
        dx[1] = -0.3 * x[1] + 0.5 * x[0] * x[1];
    };
    auto integrator = std::make_unique<diffeq::RK4Integrator<std::vector<double>>>(system);
    
    // 创建异步管理器
    StdAsyncIntegrationManager<std::vector<double>> manager(std::move(integrator));
    
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
    
    // 启动异步积分任务
    for (size_t i = 0; i < parameters.size(); ++i) {
        const auto& [alpha, beta] = parameters[i];
        
        manager.integrate_async(
            {1.0, 0.5},  // 初始状态
            0.01,        // 时间步长
            10.0,        // 结束时间
            [&analyzer, &saver, &optimizer, alpha, beta, i]
            (const std::vector<double>& final_state, double final_time) {
                
                std::cout << "任务 " << i << " 完成 (α=" << alpha << ", β=" << beta << ")" << std::endl;
                
                // 并行执行多个后处理任务
                auto analysis_future = std::async(std::launch::async, [&analyzer, &final_state, final_time]() {
                    analyzer.analyze_result(final_state, final_time);
                });
                
                auto save_future = std::async(std::launch::async, [&saver, &final_state, final_time]() {
                    saver.save_trajectory(final_state, final_time);
                });
                
                // 计算目标函数值（简化版本）
                double objective = std::abs(final_state[0] - final_state[1]);
                std::vector<double> params = {alpha, beta};
                
                auto optimize_future = std::async(std::launch::async, [&optimizer, params, objective]() {
                    optimizer.update_parameters(params, objective);
                });
                
                // 等待所有后处理任务完成
                analysis_future.wait();
                save_future.wait();
                optimize_future.wait();
            }
        );
    }
    
    // 添加一些延迟任务到队列
    manager.queue_task([]() {
        std::cout << "执行延迟任务 1: 数据清理..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    });
    
    manager.queue_task([]() {
        std::cout << "执行延迟任务 2: 结果汇总..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    });
    
    // 等待所有任务完成
    std::cout << "\n等待所有任务完成..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    manager.wait_for_all_tasks();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // 显示结果
    auto [completed, total] = manager.get_progress();
    std::cout << "\n=== 执行完成 ===" << std::endl;
    std::cout << "完成的任务: " << completed << "/" << total << std::endl;
    std::cout << "总耗时: " << duration.count() << "ms" << std::endl;
    std::cout << "平均每个任务: " << (duration.count() / total) << "ms" << std::endl;
    
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
    
    return 0;
} 