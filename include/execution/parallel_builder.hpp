#pragma once

#include <execution/parallelism_facade_clean.hpp>
#include <memory>
#include <functional>
#include <chrono>

namespace diffeq::execution {

/**
 * @brief Builder pattern for configuring complex parallel execution plans
 * 
 * This builder provides a fluent interface for setting up sophisticated
 * parallel execution configurations with hardware-specific optimizations.
 */
class ParallelExecutionBuilder {
private:
    ParallelConfig config_;
    std::vector<std::function<void()>> setup_callbacks_;
    std::vector<std::function<void()>> teardown_callbacks_;
    
public:
    ParallelExecutionBuilder() = default;
    
    // Hardware targeting
    ParallelExecutionBuilder& target_cpu() {
        config_.target = HardwareTarget::CPU_ThreadPool;
        return *this;
    }
    
    ParallelExecutionBuilder& target_gpu() {
        config_.target = HardwareTarget::GPU_CUDA;
        return *this;
    }
    
    ParallelExecutionBuilder& target_fpga() {
        config_.target = HardwareTarget::FPGA_HLS;
        return *this;
    }
    
    ParallelExecutionBuilder& auto_target() {
        config_.target = HardwareTarget::Auto;
        return *this;
    }
    
    // Parallel paradigm selection
    ParallelExecutionBuilder& use_thread_pool() {
        config_.paradigm = ParallelParadigm::ThreadPool;
        return *this;
    }
    
    ParallelExecutionBuilder& use_process_pool() {
        config_.paradigm = ParallelParadigm::ProcessPool;
        return *this;
    }
    
    ParallelExecutionBuilder& use_fibers() {
        config_.paradigm = ParallelParadigm::Fibers;
        return *this;
    }
    
    ParallelExecutionBuilder& use_simd() {
        config_.paradigm = ParallelParadigm::SIMD;
        return *this;
    }
    
    ParallelExecutionBuilder& use_task_graph() {
        config_.paradigm = ParallelParadigm::TaskGraph;
        return *this;
    }
    
    ParallelExecutionBuilder& use_pipeline() {
        config_.paradigm = ParallelParadigm::Pipeline;
        return *this;
    }
    
    // Worker configuration
    ParallelExecutionBuilder& workers(size_t count) {
        config_.max_workers = count;
        return *this;
    }
    
    ParallelExecutionBuilder& batch_size(size_t size) {
        config_.batch_size = size;
        return *this;
    }
    
    // Priority settings
    ParallelExecutionBuilder& low_priority() {
        config_.priority = Priority::Low;
        return *this;
    }
    
    ParallelExecutionBuilder& normal_priority() {
        config_.priority = Priority::Normal;
        return *this;
    }
    
    ParallelExecutionBuilder& high_priority() {
        config_.priority = Priority::High;
        return *this;
    }
    
    ParallelExecutionBuilder& critical_priority() {
        config_.priority = Priority::Critical;
        return *this;
    }
    
    ParallelExecutionBuilder& realtime_priority() {
        config_.priority = Priority::Realtime;
        return *this;
    }
    
    // Optimization flags
    ParallelExecutionBuilder& enable_load_balancing() {
        config_.enable_load_balancing = true;
        return *this;
    }
    
    ParallelExecutionBuilder& disable_load_balancing() {
        config_.enable_load_balancing = false;
        return *this;
    }
    
    ParallelExecutionBuilder& enable_numa_awareness() {
        config_.enable_numa_awareness = true;
        return *this;
    }
    
    ParallelExecutionBuilder& disable_numa_awareness() {
        config_.enable_numa_awareness = false;
        return *this;
    }
    
    // GPU-specific configuration
    ParallelExecutionBuilder& gpu_device(int device_id) {
        config_.gpu.device_id = device_id;
        return *this;
    }
    
    ParallelExecutionBuilder& gpu_block_size(size_t block_size) {
        config_.gpu.block_size = block_size;
        return *this;
    }
    
    ParallelExecutionBuilder& gpu_grid_size(size_t grid_size) {
        config_.gpu.grid_size = grid_size;
        return *this;
    }
    
    // MPI-specific configuration
    ParallelExecutionBuilder& mpi_rank(int rank) {
        config_.mpi.rank = rank;
        return *this;
    }
    
    ParallelExecutionBuilder& mpi_size(int size) {
        config_.mpi.size = size;
        return *this;
    }
    
    ParallelExecutionBuilder& enable_mpi_async() {
        config_.mpi.use_async_comm = true;
        return *this;
    }
    
    ParallelExecutionBuilder& disable_mpi_async() {
        config_.mpi.use_async_comm = false;
        return *this;
    }
    
    // Callback registration
    ParallelExecutionBuilder& on_setup(std::function<void()> callback) {
        setup_callbacks_.push_back(std::move(callback));
        return *this;
    }
    
    ParallelExecutionBuilder& on_teardown(std::function<void()> callback) {
        teardown_callbacks_.push_back(std::move(callback));
        return *this;
    }
    
    // Predefined configurations for common use cases
    ParallelExecutionBuilder& for_robotics_control() {
        return realtime_priority()
               .target_cpu()
               .use_thread_pool()
               .workers(4)
               .enable_numa_awareness()
               .batch_size(100);
    }
    
    ParallelExecutionBuilder& for_stochastic_research() {
        return normal_priority()
               .auto_target()
               .use_thread_pool()
               .workers(std::thread::hardware_concurrency())
               .enable_load_balancing()
               .batch_size(10000);
    }
    
    ParallelExecutionBuilder& for_monte_carlo() {
        return normal_priority()
               .target_gpu()
               .use_simd()
               .workers(2048)
               .batch_size(100000);
    }
    
    ParallelExecutionBuilder& for_real_time_systems() {
        return realtime_priority()
               .target_cpu()
               .use_fibers()
               .workers(2)
               .disable_load_balancing()
               .batch_size(1);
    }
    
    // Build the final facade
    std::unique_ptr<ParallelismFacade> build() {
        // Execute setup callbacks
        for (const auto& callback : setup_callbacks_) {
            callback();
        }
        
        auto facade = std::make_unique<ParallelismFacade>(config_);
        
        // Store teardown callbacks for later execution (in a real implementation)
        // For now, we'll just execute them immediately
        for (const auto& callback : teardown_callbacks_) {
            callback();
        }
        
        return facade;
    }
    
    // Alternative: Configure existing facade
    void configure(ParallelismFacade& facade) {
        for (const auto& callback : setup_callbacks_) {
            callback();
        }
        
        facade.configure(config_);
        
        for (const auto& callback : teardown_callbacks_) {
            callback();
        }
    }
    
    // Get the configuration without building
    const ParallelConfig& get_config() const {
        return config_;
    }
};

/**
 * @brief Fluent interface entry point
 */
inline ParallelExecutionBuilder parallel_execution() {
    return ParallelExecutionBuilder{};
}

/**
 * @brief Pre-configured builders for common scenarios
 */
namespace presets {

inline ParallelExecutionBuilder robotics_control() {
    return parallel_execution().for_robotics_control();
}

inline ParallelExecutionBuilder stochastic_research() {
    return parallel_execution().for_stochastic_research();
}

inline ParallelExecutionBuilder monte_carlo() {
    return parallel_execution().for_monte_carlo();
}

inline ParallelExecutionBuilder real_time_systems() {
    return parallel_execution().for_real_time_systems();
}

} // namespace presets

/**
 * @brief Pipeline execution builder for complex multi-stage computations
 */
template<typename... Stages>
class PipelineBuilder {
private:
    std::tuple<Stages...> stages_;
    ParallelConfig config_;
    
public:
    explicit PipelineBuilder(Stages... stages) : stages_(std::move(stages)...) {}
    
    template<typename NewStage>
    auto add_stage(NewStage&& stage) {
        return PipelineBuilder<Stages..., NewStage>(
            std::tuple_cat(stages_, std::make_tuple(std::forward<NewStage>(stage)))
        );
    }
    
    PipelineBuilder& configure(const ParallelConfig& config) {
        config_ = config;
        return *this;
    }
    
    template<typename Input>
    auto execute(Input&& input) {
        // This would implement the pipeline execution logic
        // For now, it's a placeholder
        return std::apply([&input](auto&&... stages) {
            return execute_pipeline(std::forward<Input>(input), stages...);
        }, stages_);
    }
    
private:
    template<typename Input, typename FirstStage, typename... RestStages>
    auto execute_pipeline(Input&& input, FirstStage&& first, RestStages&&... rest) {
        auto result = first(std::forward<Input>(input));
        if constexpr (sizeof...(rest) > 0) {
            return execute_pipeline(std::move(result), rest...);
        } else {
            return result;
        }
    }
};

template<typename... Stages>
auto make_pipeline(Stages&&... stages) {
    return PipelineBuilder<Stages...>(std::forward<Stages>(stages)...);
}

} // namespace diffeq::execution