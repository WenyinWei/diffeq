#include <diffeq.hpp>
#include <boost/asio.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/asio/steady_timer.hpp>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <mutex>

namespace asio = boost::asio;

/**
 * @brief 高级异步积分管理器 - 支持自适应参数优化
 * 
 * 这个示例展示了如何构建一个完整的参数优化系统，
 * 其中每个 ODE 计算完成后都会触发参数调整和新的计算。
 */
template<typename State>
class AdvancedAsioIntegrationManager {
private:
    asio::io_context io_context_;
    asio::thread_pool thread_pool_;
    std::unique_ptr<diffeq::core::AbstractIntegrator<State>> integrator_;
    
    // 任务管理
    std::atomic<size_t> active_tasks_{0};
    std::atomic<size_t> completed_tasks_{0};
    std::atomic<size_t> total_tasks_{0};
    
    // 参数优化状态
    std::vector<double> current_parameters_;
    std::vector<std::pair<std::vector<double>, double>> optimization_history_;
    std::mutex optimization_mutex_;
    
    // 收敛控制
    double tolerance_{1e-6};
    size_t max_iterations_{100};
    size_t current_iteration_{0};

public:
    /**
     * @brief 构造函数
     */
    AdvancedAsioIntegrationManager(std::unique_ptr<diffeq::core::AbstractIntegrator<State>> integrator,
                                  size_t thread_count = std::thread::hardware_concurrency())
        : thread_pool_(thread_count)
        , integrator_(std::move(integrator)) {
        
        if (!integrator_) {
            throw std::invalid_argument("Integrator cannot be null");
        }
    }

    /**
     * @brief 设置优化参数
     */
    void set_optimization_parameters(double tolerance, size_t max_iterations) {
        tolerance_ = tolerance;
        max_iterations_ = max_iterations;
    }

    /**
     * @brief 启动自适应参数优化
     */
    template<typename ObjectiveFunction, typename ParameterUpdateFunction>
    void optimize_parameters_async(const State& initial_state,
                                 const std::vector<double>& initial_params,
                                 ObjectiveFunction&& objective,
                                 ParameterUpdateFunction&& param_update,
                                 std::function<void(const std::vector<double>&, double)> callback = nullptr) {
        
        current_parameters_ = initial_params;
        current_iteration_ = 0;
        optimization_history_.clear();
        
        // 启动优化循环
        start_optimization_loop(initial_state, 
                              std::forward<ObjectiveFunction>(objective),
                              std::forward<ParameterUpdateFunction>(param_update),
                              std::move(callback));
    }

    /**
     * @brief 运行事件循环
     */
    void run(std::chrono::milliseconds timeout = std::chrono::milliseconds::max()) {
        if (timeout != std::chrono::milliseconds::max()) {
            asio::steady_timer timer(io_context_, timeout);
            timer.async_wait([this](const asio::error_code&) {
                io_context_.stop();
            });
        }
        
        io_context_.run();
    }

    /**
     * @brief 获取优化历史
     */
    const std::vector<std::pair<std::vector<double>, double>>& get_optimization_history() const {
        return optimization_history_;
    }

    /**
     * @brief 获取当前参数
     */
    std::vector<double> get_current_parameters() const {
        std::lock_guard<std::mutex> lock(optimization_mutex_);
        return current_parameters_;
    }

private:
    /**
     * @brief 启动优化循环
     */
    template<typename ObjectiveFunction, typename ParameterUpdateFunction>
    void start_optimization_loop(const State& initial_state,
                                ObjectiveFunction&& objective,
                                ParameterUpdateFunction&& param_update,
                                std::function<void(const std::vector<double>&, double)> callback) {
        
        asio::co_spawn(io_context_, 
            [this, initial_state, objective = std::forward<ObjectiveFunction>(objective),
             param_update = std::forward<ParameterUpdateFunction>(param_update),
             callback = std::move(callback)]() mutable -> asio::awaitable<void> {
            
            double previous_objective = std::numeric_limits<double>::max();
            
            while (current_iteration_ < max_iterations_) {
                std::cout << "优化迭代 " << current_iteration_ << std::endl;
                
                // 执行当前参数的积分
                auto [final_state, objective_value] = co_await evaluate_parameters(
                    initial_state, current_parameters_, objective);
                
                // 记录历史
                {
                    std::lock_guard<std::mutex> lock(optimization_mutex_);
                    optimization_history_.emplace_back(current_parameters_, objective_value);
                }
                
                // 检查收敛
                if (std::abs(objective_value - previous_objective) < tolerance_) {
                    std::cout << "优化收敛于迭代 " << current_iteration_ << std::endl;
                    break;
                }
                
                // 更新参数
                auto new_params = param_update(current_parameters_, objective_value, final_state);
                
                {
                    std::lock_guard<std::mutex> lock(optimization_mutex_);
                    current_parameters_ = std::move(new_params);
                }
                
                // 调用回调
                if (callback) {
                    callback(current_parameters_, objective_value);
                }
                
                previous_objective = objective_value;
                ++current_iteration_;
                
                // 短暂延迟，避免过度占用资源
                asio::steady_timer timer(io_context_, std::chrono::milliseconds(10));
                co_await timer.async_wait(asio::use_awaitable);
            }
            
            std::cout << "优化完成，总迭代次数: " << current_iteration_ << std::endl;
        }, asio::detached);
    }

    /**
     * @brief 评估参数组合
     */
    template<typename ObjectiveFunction>
    asio::awaitable<std::pair<State, double>> evaluate_parameters(
        const State& initial_state,
        const std::vector<double>& params,
        ObjectiveFunction&& objective) {
        
        return co_await asio::co_spawn(thread_pool_,
            [this, initial_state, params, objective = std::forward<ObjectiveFunction>(objective)]() 
            mutable -> asio::awaitable<std::pair<State, double>> {
            
            // 复制状态和参数
            State state = initial_state;
            std::vector<double> local_params = params;
            
            // 执行积分
            integrator_->set_time(0.0);
            integrator_->integrate(state, 0.01, 10.0);
            
            // 计算目标函数值
            double obj_value = objective(state, local_params, integrator_->current_time());
            
            co_return std::make_pair(state, obj_value);
        }, asio::use_awaitable);
    }
};

// 示例：复杂系统定义
struct LorenzSystem {
    double sigma, rho, beta;
    
    LorenzSystem(double s, double r, double b) : sigma(s), rho(r), beta(b) {}
    
    void operator()(std::vector<double>& dx, const std::vector<double>& x, double t) const {
        dx[0] = sigma * (x[1] - x[0]);
        dx[1] = x[0] * (rho - x[2]) - x[1];
        dx[2] = x[0] * x[1] - beta * x[2];
    }
};

// 示例：目标函数 - 寻找混沌行为
class ChaosObjective {
private:
    std::vector<double> target_lyapunov_exponents_;

public:
    ChaosObjective(const std::vector<double>& target_exponents = {0.9, 0.0, -14.6})
        : target_lyapunov_exponents_(target_exponents) {}
    
    double operator()(const std::vector<double>& final_state, 
                     const std::vector<double>& params, 
                     double final_time) const {
        // 简化的 Lyapunov 指数估计
        // 在实际应用中，这里会计算真正的 Lyapunov 指数
        
        double x = final_state[0], y = final_state[1], z = final_state[2];
        double sigma = params[0], rho = params[1], beta = params[2];
        
        // 基于系统参数的稳定性分析
        double stability_metric = std::abs(x) + std::abs(y) + std::abs(z);
        double parameter_balance = std::abs(sigma - 10.0) + std::abs(rho - 28.0) + std::abs(beta - 8.0/3.0);
        
        // 目标：最大化混沌行为（高稳定性指标）同时保持参数接近经典值
        return -stability_metric + 0.1 * parameter_balance;
    }
};

// 示例：参数更新策略 - 梯度下降
class GradientDescentOptimizer {
private:
    double learning_rate_;
    std::vector<double> parameter_bounds_;

public:
    GradientDescentOptimizer(double lr = 0.1, 
                            const std::vector<double>& bounds = {1.0, 50.0, 1.0, 50.0, 1.0, 10.0})
        : learning_rate_(lr), parameter_bounds_(bounds) {}
    
    std::vector<double> operator()(const std::vector<double>& current_params,
                                  double objective_value,
                                  const std::vector<double>& final_state) const {
        
        std::vector<double> new_params = current_params;
        
        // 简化的梯度估计（在实际应用中会使用更复杂的数值微分）
        for (size_t i = 0; i < new_params.size(); ++i) {
            double perturbation = 0.01 * new_params[i];
            
            // 计算数值梯度
            double gradient = estimate_gradient(i, current_params, objective_value, perturbation);
            
            // 更新参数
            new_params[i] -= learning_rate_ * gradient;
            
            // 应用边界约束
            if (i * 2 + 1 < parameter_bounds_.size()) {
                new_params[i] = std::max(parameter_bounds_[i * 2], 
                                       std::min(parameter_bounds_[i * 2 + 1], new_params[i]));
            }
        }
        
        return new_params;
    }

private:
    double estimate_gradient(size_t param_index,
                           const std::vector<double>& params,
                           double current_objective,
                           double perturbation) const {
        // 简化的梯度估计
        // 在实际应用中，这里会进行额外的函数评估
        
        // 基于参数对目标函数的影响进行启发式估计
        double param_value = params[param_index];
        double normalized_value = param_value / (param_index + 1.0);
        
        return normalized_value * std::sin(current_objective) * 0.1;
    }
};

// 示例：结果可视化器
class OptimizationVisualizer {
private:
    std::string output_dir_;
    std::mutex file_mutex_;

public:
    explicit OptimizationVisualizer(std::string output_dir = "optimization_results/")
        : output_dir_(std::move(output_dir)) {}
    
    void visualize_progress(const std::vector<double>& params, double objective_value) {
        std::lock_guard<std::mutex> lock(file_mutex_);
        
        // 创建输出目录（在实际应用中）
        // std::filesystem::create_directories(output_dir_);
        
        // 记录参数和目标值
        std::cout << "参数: [";
        for (size_t i = 0; i < params.size(); ++i) {
            std::cout << params[i];
            if (i < params.size() - 1) std::cout << ", ";
        }
        std::cout << "], 目标值: " << objective_value << std::endl;
    }
    
    void save_final_results(const std::vector<std::pair<std::vector<double>, double>>& history) {
        std::lock_guard<std::mutex> lock(file_mutex_);
        
        std::cout << "\n=== 优化历史 ===" << std::endl;
        for (size_t i = 0; i < history.size(); ++i) {
            const auto& [params, obj_value] = history[i];
            std::cout << "迭代 " << i << ": 目标值 = " << obj_value 
                      << ", 参数 = [" << params[0] << ", " << params[1] << ", " << params[2] << "]" << std::endl;
        }
    }
};

int main() {
    std::cout << "=== 高级 Boost.Asio 积分器集成示例 ===" << std::endl;
    std::cout << "自适应参数优化系统" << std::endl;
    
    // 创建积分器
    auto integrator = std::make_unique<diffeq::RK4Integrator<std::vector<double>>>();
    
    // 创建高级异步管理器
    AdvancedAsioIntegrationManager<std::vector<double>> manager(std::move(integrator), 4);
    
    // 设置优化参数
    manager.set_optimization_parameters(1e-4, 20);  // 容差和最大迭代次数
    
    // 创建优化组件
    ChaosObjective objective;
    GradientDescentOptimizer optimizer(0.05);
    OptimizationVisualizer visualizer;
    
    // 初始参数（Lorenz 系统的经典参数附近）
    std::vector<double> initial_params = {8.0, 25.0, 2.5};  // sigma, rho, beta
    std::vector<double> initial_state = {1.0, 1.0, 1.0};    // x, y, z
    
    std::cout << "\n开始参数优化..." << std::endl;
    std::cout << "初始参数: [" << initial_params[0] << ", " << initial_params[1] << ", " << initial_params[2] << "]" << std::endl;
    
    // 启动异步优化
    manager.optimize_parameters_async(
        initial_state,
        initial_params,
        objective,
        optimizer,
        [&visualizer](const std::vector<double>& params, double obj_value) {
            visualizer.visualize_progress(params, obj_value);
        }
    );
    
    // 运行事件循环
    auto start_time = std::chrono::high_resolution_clock::now();
    manager.run();
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // 显示结果
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "\n=== 优化完成 ===" << std::endl;
    std::cout << "总耗时: " << duration.count() << "ms" << std::endl;
    
    // 显示最终结果
    auto final_params = manager.get_current_parameters();
    std::cout << "最终参数: [" << final_params[0] << ", " << final_params[1] << ", " << final_params[2] << "]" << std::endl;
    
    // 保存优化历史
    const auto& history = manager.get_optimization_history();
    visualizer.save_final_results(history);
    
    // 验证最终结果
    std::cout << "\n=== 验证最终参数 ===" << std::endl;
    LorenzSystem final_system(final_params[0], final_params[1], final_params[2]);
    std::cout << "Lorenz 系统参数: σ=" << final_params[0] 
              << ", ρ=" << final_params[1] 
              << ", β=" << final_params[2] << std::endl;
    
    return 0;
} 