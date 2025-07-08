#include <diffeq.hpp>
#include <boost/asio.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <random>

namespace asio = boost::asio;

/**
 * @brief 使用 boost.asio 的异步积分器包装器
 * 
 * 这个示例展示了如何将 boost.asio 与我们的积分器结合使用，
 * 实现 ODE 计算完成后的异步任务处理，如数据分析、参数调整、轨迹保存等。
 * 
 * 设计理念：
 * - 利用 boost.asio 的成熟异步设施，避免重复发明轮子
 * - 专注于 ODE 计算完成后的任务编排，而不是积分器内部的异步化
 * - 支持高并行度的 ODE 运算后的数据分析流程
 */
template<typename State>
class AsioIntegrationManager {
private:
    asio::io_context io_context_;
    asio::thread_pool thread_pool_;
    std::unique_ptr<diffeq::core::AbstractIntegrator<State>> integrator_;
    
    // 任务队列和状态管理
    std::vector<std::future<void>> pending_tasks_;
    std::atomic<size_t> completed_integrations_{0};
    std::atomic<size_t> total_integrations_{0};

public:
    /**
     * @brief 构造函数
     * @param integrator 积分器实例
     * @param thread_count 线程池大小
     */
    AsioIntegrationManager(std::unique_ptr<diffeq::core::AbstractIntegrator<State>> integrator, 
                          size_t thread_count = std::thread::hardware_concurrency())
        : thread_pool_(thread_count)
        , integrator_(std::move(integrator)) {
        
        if (!integrator_) {
            throw std::invalid_argument("Integrator cannot be null");
        }
    }

    /**
     * @brief 析构函数 - 确保所有任务完成
     */
    ~AsioIntegrationManager() {
        wait_for_all_tasks();
        thread_pool_.join();
    }

    /**
     * @brief 异步执行积分任务
     * @param initial_state 初始状态
     * @param dt 时间步长
     * @param end_time 结束时间
     * @param post_integration_task 积分完成后的回调任务
     */
    template<typename PostTask>
    void integrate_async(State initial_state, 
                        typename diffeq::core::AbstractIntegrator<State>::time_type dt,
                        typename diffeq::core::AbstractIntegrator<State>::time_type end_time,
                        PostTask&& post_integration_task) {
        
        ++total_integrations_;
        
        // 使用 asio::co_spawn 启动协程
        asio::co_spawn(io_context_, 
            [this, initial_state = std::move(initial_state), dt, end_time, 
             task = std::forward<PostTask>(post_integration_task)]() mutable -> asio::awaitable<void> {
                
            try {
                // 在线程池中执行积分计算
                auto integration_result = co_await asio::co_spawn(thread_pool_, 
                    [this, &initial_state, dt, end_time]() -> asio::awaitable<std::pair<State, double>> {
                        
                    // 执行积分
                    integrator_->set_time(0.0);
                    integrator_->integrate(initial_state, dt, end_time);
                    
                    co_return std::make_pair(initial_state, integrator_->current_time());
                }, asio::use_awaitable);

                // 积分完成，执行后续任务
                co_await asio::co_spawn(thread_pool_, 
                    [task = std::move(task), state = std::move(integration_result.first), 
                     final_time = integration_result.second]() mutable -> asio::awaitable<void> {
                    
                    // 执行用户定义的后处理任务
                    task(state, final_time);
                    co_return;
                }, asio::use_awaitable);

                ++completed_integrations_;
                
            } catch (const std::exception& e) {
                std::cerr << "Integration task failed: " << e.what() << std::endl;
            }
        }, asio::detached);
    }

    /**
     * @brief 批量执行多个积分任务
     * @param tasks 任务列表，每个任务包含初始状态、积分参数和后处理函数
     */
    template<typename TaskList>
    void integrate_batch_async(TaskList&& tasks) {
        for (auto& task : tasks) {
            integrate_async(std::move(task.initial_state), 
                          task.dt, task.end_time, 
                          std::move(task.post_task));
        }
    }

    /**
     * @brief 运行事件循环
     * @param timeout 超时时间（可选）
     */
    void run(std::chrono::milliseconds timeout = std::chrono::milliseconds::max()) {
        if (timeout != std::chrono::milliseconds::max()) {
            // 设置超时
            asio::steady_timer timer(io_context_, timeout);
            timer.async_wait([this](const asio::error_code&) {
                io_context_.stop();
            });
        }
        
        io_context_.run();
    }

    /**
     * @brief 等待所有任务完成
     */
    void wait_for_all_tasks() {
        while (completed_integrations_.load() < total_integrations_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    /**
     * @brief 获取完成统计
     */
    std::pair<size_t, size_t> get_progress() const {
        return {completed_integrations_.load(), total_integrations_.load()};
    }

    /**
     * @brief 重置统计
     */
    void reset_stats() {
        completed_integrations_.store(0);
        total_integrations_.store(0);
    }
};

// 示例：参数化 ODE 系统
struct ParameterizedSystem {
    double alpha;
    double beta;
    
    ParameterizedSystem(double a, double b) : alpha(a), beta(b) {}
    
    void operator()(std::vector<double>& dx, const std::vector<double>& x, double t) const {
        dx[0] = alpha * x[0] - beta * x[0] * x[1];  // 捕食者-被捕食者模型
        dx[1] = -beta * x[1] + alpha * x[0] * x[1];
    }
};

// 示例：数据分析任务
class DataAnalyzer {
private:
    std::vector<std::pair<std::vector<double>, double>> trajectory_data_;
    std::mutex data_mutex_;

public:
    /**
     * @brief 分析轨迹数据并调整参数
     */
    void analyze_and_adjust_parameters(const std::vector<double>& final_state, double final_time) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        // 模拟数据分析
        std::cout << "分析轨迹数据: 最终状态 = [" 
                  << final_state[0] << ", " << final_state[1] 
                  << "], 时间 = " << final_time << std::endl;
        
        // 基于分析结果调整参数（这里只是示例）
        double stability_metric = std::abs(final_state[0] - final_state[1]);
        std::cout << "稳定性指标: " << stability_metric << std::endl;
        
        trajectory_data_.emplace_back(final_state, final_time);
    }
    
    /**
     * @brief 获取分析结果
     */
    const std::vector<std::pair<std::vector<double>, double>>& get_trajectory_data() const {
        return trajectory_data_;
    }
};

// 示例：轨迹保存任务
class TrajectorySaver {
private:
    std::string filename_prefix_;
    std::atomic<size_t> save_count_{0};

public:
    explicit TrajectorySaver(std::string prefix = "trajectory_") 
        : filename_prefix_(std::move(prefix)) {}
    
    /**
     * @brief 保存轨迹数据
     */
    void save_trajectory(const std::vector<double>& final_state, double final_time) {
        auto count = ++save_count_;
        std::string filename = filename_prefix_ + std::to_string(count) + ".dat";
        
        // 模拟文件保存操作
        std::cout << "保存轨迹到文件: " << filename 
                  << " (状态: [" << final_state[0] << ", " << final_state[1] 
                  << "], 时间: " << final_time << ")" << std::endl;
        
        // 这里可以实际写入文件
        // std::ofstream file(filename);
        // file << final_time << " " << final_state[0] << " " << final_state[1] << "\n";
    }
};

int main() {
    std::cout << "=== Boost.Asio 与积分器集成示例 ===" << std::endl;
    
    // 创建积分器
    auto integrator = std::make_unique<diffeq::RK4Integrator<std::vector<double>>>();
    
    // 创建异步管理器
    AsioIntegrationManager<std::vector<double>> manager(std::move(integrator), 4);
    
    // 创建后处理组件
    DataAnalyzer analyzer;
    TrajectorySaver saver("async_traj_");
    
    // 定义不同的参数组合
    std::vector<std::pair<double, double>> parameter_sets = {
        {0.5, 0.3}, {0.8, 0.2}, {0.3, 0.7}, {0.6, 0.4},
        {0.4, 0.6}, {0.7, 0.3}, {0.2, 0.8}, {0.9, 0.1}
    };
    
    std::cout << "\n启动 " << parameter_sets.size() << " 个异步积分任务..." << std::endl;
    
    // 启动多个异步积分任务
    for (size_t i = 0; i < parameter_sets.size(); ++i) {
        const auto& [alpha, beta] = parameter_sets[i];
        
        // 设置系统
        ParameterizedSystem system(alpha, beta);
        manager.integrate_async(
            {1.0, 0.5},  // 初始状态
            0.01,        // 时间步长
            10.0,        // 结束时间
            [&analyzer, &saver, alpha, beta, i](const std::vector<double>& final_state, double final_time) {
                std::cout << "任务 " << i << " 完成 (α=" << alpha << ", β=" << beta << ")" << std::endl;
                
                // 并行执行数据分析和轨迹保存
                analyzer.analyze_and_adjust_parameters(final_state, final_time);
                saver.save_trajectory(final_state, final_time);
            }
        );
    }
    
    // 运行事件循环
    std::cout << "\n运行事件循环..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    manager.run();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // 显示结果
    auto [completed, total] = manager.get_progress();
    std::cout << "\n=== 执行完成 ===" << std::endl;
    std::cout << "完成的任务: " << completed << "/" << total << std::endl;
    std::cout << "总耗时: " << duration.count() << "ms" << std::endl;
    std::cout << "平均每个任务: " << (duration.count() / total) << "ms" << std::endl;
    
    // 显示分析结果
    const auto& trajectory_data = analyzer.get_trajectory_data();
    std::cout << "\n收集的轨迹数据点: " << trajectory_data.size() << std::endl;
    
    return 0;
} 