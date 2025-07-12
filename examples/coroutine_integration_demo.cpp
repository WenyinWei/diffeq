#define NOMINMAX
#include <diffeq.hpp>
#include <coroutine>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <optional>
#include <memory>

#ifdef _WIN32
#include <windows.h>
#endif

/**
 * @brief C++20 协程与 diffeq 库集成示例
 * 
 * 这个示例展示了如何使用 C++20 协程特性与微分方程积分器结合，
 * 实现更细粒度的 CPU 运行控制和任务调度。
 * 
 * 主要特性：
 * - 协程化的积分步进，可以暂停和恢复
 * - 细粒度的进度报告和控制
 * - 协作式多任务处理
 * - 零拷贝的状态传递
 */

// ============================================================================
// 协程基础设施
// ============================================================================

/**
 * @brief 积分任务的协程返回类型
 */
template<typename State>
struct IntegrationTask {
    struct promise_type {
        State current_state;
        double current_time{0.0};
        std::exception_ptr exception;
        
        IntegrationTask get_return_object() {
            return IntegrationTask{
                std::coroutine_handle<promise_type>::from_promise(*this)
            };
        }
        
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        
        void unhandled_exception() {
            exception = std::current_exception();
        }
        
        void return_void() {}
        
        // 允许协程 co_yield 状态和时间
        std::suspend_always yield_value(std::pair<const State&, double> value) {
            current_state = value.first;
            current_time = value.second;
            return {};
        }
    };
    
    using handle_type = std::coroutine_handle<promise_type>;
    handle_type coro;
    
    explicit IntegrationTask(handle_type h) : coro(h) {}
    
    ~IntegrationTask() {
        if (coro) {
            coro.destroy();
        }
    }
    
    // 移动构造和赋值
    IntegrationTask(IntegrationTask&& other) noexcept 
        : coro(std::exchange(other.coro, {})) {}
    
    IntegrationTask& operator=(IntegrationTask&& other) noexcept {
        if (this != &other) {
            if (coro) coro.destroy();
            coro = std::exchange(other.coro, {});
        }
        return *this;
    }
    
    // 禁用拷贝
    IntegrationTask(const IntegrationTask&) = delete;
    IntegrationTask& operator=(const IntegrationTask&) = delete;
    
    // 恢复协程执行
    bool resume() {
        if (!coro || coro.done()) return false;
        coro.resume();
        return !coro.done();
    }
    
    // 检查是否完成
    bool done() const {
        return !coro || coro.done();
    }
    
    // 获取当前状态
    std::pair<State, double> get_current() const {
        if (coro) {
            return {coro.promise().current_state, coro.promise().current_time};
        }
        throw std::runtime_error("No current state available");
    }
    
    // 检查异常
    void check_exception() {
        if (coro && coro.promise().exception) {
            std::rethrow_exception(coro.promise().exception);
        }
    }
    
    // 使 IntegrationTask 可等待（awaitable）
    bool await_ready() const noexcept {
        return done();
    }
    
    void await_suspend(std::coroutine_handle<> h) {
        // 在另一个线程中运行任务直到完成，然后恢复等待的协程
        std::thread([this, h]() {
            while (!done()) {
                resume();
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
            }
            h.resume();
        }).detach();
    }
    
    std::pair<State, double> await_resume() {
        check_exception();
        if (coro) {
            return {coro.promise().current_state, coro.promise().current_time};
        }
        throw std::runtime_error("Coroutine not available");
    }
};

/**
 * @brief 可等待的延迟对象，用于协程中的定时暂停
 */
struct TimedSuspend {
    std::chrono::milliseconds delay;
    
    bool await_ready() const noexcept { return delay.count() <= 0; }
    
    void await_suspend(std::coroutine_handle<> h) const {
        std::thread([h, this]() {
            std::this_thread::sleep_for(delay);
            h.resume();
        }).detach();
    }
    
    void await_resume() const noexcept {}
};

// ============================================================================
// 协程化的积分器包装
// ============================================================================

/**
 * @brief 将积分器包装为协程，支持细粒度控制
 */
template<typename State>
class CoroutineIntegrator {
private:
    std::unique_ptr<diffeq::core::AbstractIntegrator<State>> integrator_;
    
public:
    explicit CoroutineIntegrator(
        std::unique_ptr<diffeq::core::AbstractIntegrator<State>> integrator)
        : integrator_(std::move(integrator)) {}
    
    /**
     * @brief 协程化的积分，每步都可以暂停
     * @param initial_state 初始状态
     * @param dt 时间步长
     * @param end_time 结束时间
     * @param yield_interval 产生中间结果的步数间隔
     */
    IntegrationTask<State> integrate_coro(
        State initial_state,
        typename diffeq::core::AbstractIntegrator<State>::time_type dt,
        typename diffeq::core::AbstractIntegrator<State>::time_type end_time,
        size_t yield_interval = 10) {
        
        State state = std::move(initial_state);
        double current_time = 0.0;
        integrator_->set_time(current_time);
        size_t step_count = 0;
        
        while (current_time < end_time) {
            // 执行一步积分
            auto step_dt = std::min(dt, end_time - current_time);
            integrator_->step(state, step_dt);
            current_time += step_dt;  // 手动更新时间
            integrator_->set_time(current_time);  // 同步积分器时间
            step_count++;
            
            // 每隔一定步数，暂停并返回当前状态
            if (step_count % yield_interval == 0) {
                co_yield std::make_pair(std::cref(state), current_time);
            }
        }
        
        // 返回最终状态
        co_yield std::make_pair(std::cref(state), current_time);
    }
    
    /**
     * @brief 带进度回调的协程积分
     */
    template<typename ProgressCallback>
    IntegrationTask<State> integrate_with_progress(
        State initial_state,
        typename diffeq::core::AbstractIntegrator<State>::time_type dt,
        typename diffeq::core::AbstractIntegrator<State>::time_type end_time,
        ProgressCallback&& callback) {
        
        State state = std::move(initial_state);
        double current_time = 0.0;
        integrator_->set_time(current_time);
        
        while (current_time < end_time) {
            // 执行积分步
            auto step_dt = std::min(dt, end_time - current_time);
            integrator_->step(state, step_dt);
            current_time += step_dt;  // 手动更新时间
            integrator_->set_time(current_time);  // 同步积分器时间
            
            // 调用进度回调
            double progress = current_time / end_time;
            bool should_continue = callback(state, current_time, progress);
            
            if (!should_continue) {
                break;  // 用户请求停止
            }
            
            // 让出控制权
            co_yield std::make_pair(std::cref(state), current_time);
        }
    }
};

// ============================================================================
// 协程任务调度器
// ============================================================================

/**
 * @brief 简单的协程任务调度器
 */
class CoroutineScheduler {
private:
    struct Task {
        std::function<bool()> resume_func;
        std::string name;
        std::chrono::steady_clock::time_point last_run;
        std::chrono::milliseconds interval;
    };
    
    std::vector<Task> tasks_;
    std::mutex mutex_;
    
public:
    /**
     * @brief 添加一个协程任务
     */
    template<typename State>
    void add_task(IntegrationTask<State>&& task, 
                 const std::string& name,
                 std::chrono::milliseconds interval = std::chrono::milliseconds{0}) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 捕获任务的共享指针，确保生命周期
        auto task_ptr = std::make_shared<IntegrationTask<State>>(std::move(task));
        
        tasks_.push_back({
            [task_ptr]() { return task_ptr->resume(); },
            name,
            std::chrono::steady_clock::now(),
            interval
        });
    }
    
    /**
     * @brief 运行调度器
     * @param duration 运行时长
     */
    void run(std::chrono::milliseconds duration) {
        auto end_time = std::chrono::steady_clock::now() + duration;
        
        while (std::chrono::steady_clock::now() < end_time) {
            std::vector<Task> active_tasks;
            
            {
                std::lock_guard<std::mutex> lock(mutex_);
                // 移除已完成的任务，保留活跃任务
                for (auto& task : tasks_) {
                    auto now = std::chrono::steady_clock::now();
                    if (now - task.last_run >= task.interval) {
                        if (task.resume_func()) {
                            task.last_run = now;
                            active_tasks.push_back(task);
                        } else {
                            std::cout << "任务 '" << task.name << "' 已完成" << std::endl;
                        }
                    } else {
                        active_tasks.push_back(task);
                    }
                }
                tasks_ = std::move(active_tasks);
            }
            
            // 短暂休眠，避免忙等待
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
    }
    
    /**
     * @brief 获取活跃任务数
     */
    size_t active_task_count() {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_.size();
    }
};

// ============================================================================
// 示例：多尺度积分
// ============================================================================

/**
 * @brief 多尺度系统的协程积分示例
 * 
 * 展示如何使用协程处理具有不同时间尺度的耦合系统
 */
IntegrationTask<std::vector<double>> multiscale_integration_coro(
    double epsilon = 0.01) {
    
    // 快-慢耦合系统
    auto system = [epsilon](double t, const std::vector<double>& x, 
                           std::vector<double>& dx) {
        // 慢变量
        dx[0] = -x[0] + x[1];
        // 快变量
        dx[1] = -(1.0/epsilon) * (x[1] - x[0]*x[0]);
    };
    
    // 创建积分器
    auto integrator = std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(system);
    CoroutineIntegrator<std::vector<double>> coro_integrator(std::move(integrator));
    
    // 初始条件
    std::vector<double> state = {1.0, 0.0};
    
    std::cout << "\n=== 多尺度系统积分 (ε = " << epsilon << ") ===" << std::endl;
    
    // 使用自适应步长，协程每10步返回一次
    auto task = coro_integrator.integrate_coro(state, 0.001, 5.0, 10);
    
    size_t yield_count = 0;
    while (!task.done()) {
        task.resume();
        
        if (!task.done()) {
            auto [current_state, current_time] = task.get_current();
            yield_count++;
            
            // 每50次yield打印一次状态
            if (yield_count % 50 == 0) {
                std::cout << "t = " << current_time 
                          << ", 慢变量 = " << current_state[0]
                          << ", 快变量 = " << current_state[1] << std::endl;
            }
            
            // 模拟其他计算
            co_await TimedSuspend{std::chrono::milliseconds{1}};
        }
    }
    
    std::cout << "多尺度积分完成，共 yield " << yield_count << " 次" << std::endl;
}

// ============================================================================
// 示例：参数扫描与动态调度
// ============================================================================

/**
 * @brief 使用协程进行参数扫描，动态调整计算资源
 */
template<typename State>
IntegrationTask<State> parameter_scan_coro(
    double param,
    std::function<void(double, const State&, double)> result_handler) {
    
    // 参数化的 Van der Pol 振荡器
    auto system = [param](double t, const std::vector<double>& x, 
                         std::vector<double>& dx) {
        dx[0] = x[1];
        dx[1] = param * (1 - x[0]*x[0]) * x[1] - x[0];
    };
    
    auto integrator = std::make_unique<diffeq::RK4Integrator<std::vector<double>>>(system);
    CoroutineIntegrator<std::vector<double>> coro_integrator(std::move(integrator));
    
    State state = {2.0, 0.0};
    
    // 带进度监控的积分
    auto task = coro_integrator.integrate_with_progress(
        state, 0.01, 20.0,
        [param](const auto& s, double t, double progress) {
            // 仅在关键时刻输出
            if (static_cast<int>(progress * 100) % 25 == 0 && 
                static_cast<int>(progress * 100) % 25 < 1) {
                std::cout << "参数 " << param << " 的积分进度: " 
                          << static_cast<int>(progress * 100) << "%" << std::endl;
            }
            return true;  // 继续积分
        }
    );
    
    // 执行积分
    while (!task.done()) {
        task.resume();
        
        if (!task.done()) {
            auto [current_state, current_time] = task.get_current();
            
            // 让其他协程有机会运行
            co_await std::suspend_always{};
        }
    }
    
    // 获取最终结果
    auto [final_state, final_time] = task.get_current();
    result_handler(param, final_state, final_time);
}

// ============================================================================
// 主程序
// ============================================================================

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::cout << "=== C++20 协程与 diffeq 库集成示例 ===" << std::endl;
    std::cout << "展示协程在细粒度 CPU 运行控制上的优势\n" << std::endl;
    
    // 1. 基本协程积分示例
    std::cout << "1. 基本协程积分示例" << std::endl;
    {
        // Lorenz 系统
        auto lorenz = [](double t, const std::vector<double>& x, 
                        std::vector<double>& dx) {
            const double sigma = 10.0, rho = 28.0, beta = 8.0/3.0;
            dx[0] = sigma * (x[1] - x[0]);
            dx[1] = x[0] * (rho - x[2]) - x[1];
            dx[2] = x[0] * x[1] - beta * x[2];
        };
        
        auto integrator = std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(lorenz);
        CoroutineIntegrator<std::vector<double>> coro_integrator(std::move(integrator));
        
        std::vector<double> initial_state = {1.0, 1.0, 1.0};
        
        // 创建协程任务
        auto task = coro_integrator.integrate_coro(initial_state, 0.01, 2.0, 20);
        
        std::cout << "开始 Lorenz 系统积分..." << std::endl;
        size_t step_count = 0;
        
        while (!task.done()) {
            task.resume();
            step_count++;
            
            if (!task.done() && step_count % 5 == 0) {
                try {
                    auto [state, time] = task.get_current();
                    std::cout << "  t = " << time 
                              << ", ||x|| = " << std::sqrt(state[0]*state[0] + 
                                                          state[1]*state[1] + 
                                                          state[2]*state[2]) 
                              << std::endl;
                } catch (const std::exception& e) {
                    std::cout << "  获取 Lorenz 状态失败: " << e.what() << std::endl;
                }
            }
        }
        
        std::cout << "Lorenz 积分完成，共暂停/恢复 " << step_count << " 次\n" << std::endl;
    }
    
    // 2. 参数扫描示例（简化版）
    std::cout << "2. 参数扫描示例" << std::endl;
    {
        std::vector<double> parameters = {0.5, 1.0, 2.0};
        std::vector<std::pair<double, std::vector<double>>> results;
        
        for (double param : parameters) {
            std::cout << "开始参数 μ = " << param << " 的积分..." << std::endl;
            
            auto system = [param](double t, const std::vector<double>& x, 
                                 std::vector<double>& dx) {
                dx[0] = x[1];
                dx[1] = param * (1 - x[0]*x[0]) * x[1] - x[0];
            };
            
            auto integrator = std::make_unique<diffeq::RK4Integrator<std::vector<double>>>(system);
            CoroutineIntegrator<std::vector<double>> coro_integrator(std::move(integrator));
            
            std::vector<double> state = {2.0, 0.0};
            auto task = coro_integrator.integrate_coro(state, 0.05, 10.0, 50);
            
            // 逐步执行并监控进度
            size_t step_count = 0;
            while (!task.done()) {
                task.resume();
                step_count++;
                
                if (!task.done() && step_count % 20 == 0) {
                    try {
                        auto [current_state, current_time] = task.get_current();
                        std::cout << "  μ = " << param 
                                  << ", t = " << current_time 
                                  << ", x = [" << current_state[0] << ", " << current_state[1] << "]" 
                                  << std::endl;
                    } catch (const std::exception& e) {
                        std::cout << "  获取状态失败: " << e.what() << std::endl;
                    }
                }
            }
            
            try {
                auto [final_state, final_time] = task.get_current();
                results.emplace_back(param, final_state);
                std::cout << "参数 μ = " << param << " 积分完成" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "参数 μ = " << param << " 积分失败: " << e.what() << std::endl;
            }
        }
        
        std::cout << "\n参数扫描结果：" << std::endl;
        for (const auto& [param, state] : results) {
            std::cout << "  μ = " << param 
                      << ", 最终状态 = [" << state[0] << ", " << state[1] << "]" 
                      << std::endl;
        }
    }
    
    // 3. 带进度监控的协程积分
    std::cout << "\n3. 带进度监控的协程积分" << std::endl;
    {
        // 阻尼振荡器
        auto damped_oscillator = [](double t, const std::vector<double>& x, 
                                   std::vector<double>& dx) {
            double omega = 2.0, gamma = 0.1;
            dx[0] = x[1];
            dx[1] = -omega*omega*x[0] - 2*gamma*x[1];
        };
        
        auto integrator = std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(damped_oscillator);
        CoroutineIntegrator<std::vector<double>> coro_integrator(std::move(integrator));
        
        std::vector<double> state = {1.0, 0.0};
        
        auto task = coro_integrator.integrate_with_progress(
            state, 0.01, 10.0,
            [](const auto& s, double t, double progress) {
                // 每25%进度报告一次
                int progress_percent = static_cast<int>(progress * 100);
                if (progress_percent % 25 == 0 && progress_percent > 0) {
                    std::cout << "  进度: " << progress_percent 
                              << "%, t = " << t 
                              << ", 能量 = " << 0.5 * (s[0]*s[0] + s[1]*s[1])
                              << std::endl;
                }
                return true;  // 继续积分
            }
        );
        
        std::cout << "开始阻尼振荡器积分..." << std::endl;
        while (!task.done()) {
            task.resume();
        }
        
        auto [final_state, final_time] = task.get_current();
        std::cout << "阻尼振荡器积分完成，最终能量 = " 
                  << 0.5 * (final_state[0]*final_state[0] + final_state[1]*final_state[1])
                  << std::endl;
    }
    
    // 4. 协程的细粒度控制示例
    std::cout << "\n4. 协程的细粒度控制示例" << std::endl;
    {
        // 二体问题（简化的轨道力学）
        auto orbital_system = [](double t, const std::vector<double>& x, 
                                std::vector<double>& dx) {
            double mu = 1.0;  // 引力参数
            double r = std::sqrt(x[0]*x[0] + x[1]*x[1]);
            double r3 = r*r*r;
            
            dx[0] = x[2];  // vx
            dx[1] = x[3];  // vy  
            dx[2] = -mu * x[0] / r3;  // ax
            dx[3] = -mu * x[1] / r3;  // ay
        };
        
        auto integrator = std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(orbital_system);
        CoroutineIntegrator<std::vector<double>> coro_integrator(std::move(integrator));
        
        // 椭圆轨道初始条件
        std::vector<double> state = {1.0, 0.0, 0.0, 0.8};
        
        auto task = coro_integrator.integrate_coro(state, 0.01, 6.28, 25);  // 一个轨道周期
        
        std::cout << "开始轨道积分（每25步暂停一次）..." << std::endl;
        size_t resume_count = 0;
        
        while (!task.done()) {
            task.resume();
            resume_count++;
            
            if (!task.done()) {
                auto [current_state, current_time] = task.get_current();
                double r = std::sqrt(current_state[0]*current_state[0] + current_state[1]*current_state[1]);
                double v = std::sqrt(current_state[2]*current_state[2] + current_state[3]*current_state[3]);
                double energy = 0.5 * v*v - 1.0/r;  // 比能量
                
                std::cout << "  第 " << resume_count << " 次恢复: t = " << current_time 
                          << ", r = " << r 
                          << ", 能量 = " << energy << std::endl;
                
                // 演示协程的暂停特性
                std::this_thread::sleep_for(std::chrono::milliseconds{10});
            }
        }
        
        auto [final_state, final_time] = task.get_current();
        std::cout << "轨道积分完成，共恢复 " << resume_count << " 次" << std::endl;
        std::cout << "最终位置: [" << final_state[0] << ", " << final_state[1] << "]" << std::endl;
    }
    
    std::cout << "\n=== 协程集成演示完成 ===" << std::endl;
    std::cout << "关键优势：" << std::endl;
    std::cout << "- 细粒度的执行控制" << std::endl;
    std::cout << "- 协作式多任务处理" << std::endl;
    std::cout << "- 零开销的状态保存和恢复" << std::endl;
    std::cout << "- 与标准库的无缝集成" << std::endl;
    
    return 0;
} 