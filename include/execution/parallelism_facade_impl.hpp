#pragma once

#include <execution/parallelism_facade.hpp>
#include <thread>
#include <iostream>
#include <algorithm>

namespace diffeq::execution {

// CPUStrategy implementation
CPUStrategy::CPUStrategy(const ParallelConfig& config) 
    : config_(config) {
    
    // For now, we'll use basic threading instead of AdvancedExecutor
    // to avoid dependency issues
}

// GPUStrategy implementation  
GPUStrategy::GPUStrategy(const ParallelConfig& config) 
    : config_(config), cuda_available_(false), opencl_available_(false) {
    check_gpu_availability();
}

size_t GPUStrategy::get_max_concurrency() const {
    // Return a typical GPU concurrency estimate
    return cuda_available_ ? 2048 : (opencl_available_ ? 1024 : 1);
}

void GPUStrategy::check_gpu_availability() {
    // In a full implementation, this would check for CUDA/OpenCL runtime
    // For now, we'll assume they're not available
    cuda_available_ = false;
    opencl_available_ = false;
    
    // Placeholder for actual GPU detection:
    // cuda_available_ = check_cuda_runtime();
    // opencl_available_ = check_opencl_runtime();
}

// FPGAStrategy implementation
FPGAStrategy::FPGAStrategy(const ParallelConfig& config) 
    : config_(config), fpga_available_(false) {
    check_fpga_availability();
}

void FPGAStrategy::check_fpga_availability() {
    // In a full implementation, this would check for FPGA hardware/drivers
    fpga_available_ = false;
    
    // Placeholder for actual FPGA detection:
    // fpga_available_ = check_fpga_devices();
}

// ExecutionStrategyFactory implementation
std::unique_ptr<ExecutionStrategy> ExecutionStrategyFactory::create(const ParallelConfig& config) {
    switch (config.target) {
        case HardwareTarget::CPU_Sequential:
        case HardwareTarget::CPU_OpenMP:
        case HardwareTarget::CPU_MPI:
        case HardwareTarget::CPU_ThreadPool:
            return std::make_unique<CPUStrategy>(config);
            
        case HardwareTarget::GPU_CUDA:
        case HardwareTarget::GPU_OpenCL:
            return std::make_unique<GPUStrategy>(config);
            
        case HardwareTarget::FPGA_HLS:
            return std::make_unique<FPGAStrategy>(config);
            
        case HardwareTarget::Auto:
            return create_auto(config);
            
        default:
            return std::make_unique<CPUStrategy>(config);
    }
}

std::unique_ptr<ExecutionStrategy> ExecutionStrategyFactory::create_auto(const ParallelConfig& base_config) {
    // Try GPU first (highest performance potential)
    auto gpu_config = base_config;
    gpu_config.target = HardwareTarget::GPU_CUDA;
    auto gpu_strategy = std::make_unique<GPUStrategy>(gpu_config);
    if (gpu_strategy->is_available()) {
        return gpu_strategy;
    }
    
    // Try OpenCL GPU
    gpu_config.target = HardwareTarget::GPU_OpenCL;
    gpu_strategy = std::make_unique<GPUStrategy>(gpu_config);
    if (gpu_strategy->is_available()) {
        return gpu_strategy;
    }
    
    // Try FPGA
    auto fpga_config = base_config;
    fpga_config.target = HardwareTarget::FPGA_HLS;
    auto fpga_strategy = std::make_unique<FPGAStrategy>(fpga_config);
    if (fpga_strategy->is_available()) {
        return fpga_strategy;
    }
    
    // Fallback to CPU
    auto cpu_config = base_config;
    cpu_config.target = HardwareTarget::CPU_ThreadPool;
    return std::make_unique<CPUStrategy>(cpu_config);
}

std::vector<HardwareTarget> ExecutionStrategyFactory::get_available_targets() {
    std::vector<HardwareTarget> targets;
    
    // CPU is always available
    targets.push_back(HardwareTarget::CPU_Sequential);
    targets.push_back(HardwareTarget::CPU_ThreadPool);
    
    // Check GPU availability
    GPUStrategy gpu_test({});
    if (gpu_test.is_available()) {
        targets.push_back(HardwareTarget::GPU_CUDA);
        targets.push_back(HardwareTarget::GPU_OpenCL);
    }
    
    // Check FPGA availability
    FPGAStrategy fpga_test({});
    if (fpga_test.is_available()) {
        targets.push_back(HardwareTarget::FPGA_HLS);
    }
    
    return targets;
}

bool ExecutionStrategyFactory::is_target_available(HardwareTarget target) {
    auto available = get_available_targets();
    return std::find(available.begin(), available.end(), target) != available.end();
}

// ParallelismFacade implementation
ParallelismFacade::ParallelismFacade(const ParallelConfig& config) 
    : config_(config) {
    strategy_ = ExecutionStrategyFactory::create(config_);
}

void ParallelismFacade::set_hardware_target(HardwareTarget target) {
    config_.target = target;
    strategy_ = ExecutionStrategyFactory::create(config_);
}

void ParallelismFacade::set_parallel_paradigm(ParallelParadigm paradigm) {
    config_.paradigm = paradigm;
    strategy_ = ExecutionStrategyFactory::create(config_);
}

void ParallelismFacade::set_max_workers(size_t workers) {
    config_.max_workers = workers;
    strategy_ = ExecutionStrategyFactory::create(config_);
}

void ParallelismFacade::configure(const ParallelConfig& config) {
    config_ = config;
    strategy_ = ExecutionStrategyFactory::create(config_);
}

HardwareTarget ParallelismFacade::get_current_target() const {
    return strategy_->get_target();
}

size_t ParallelismFacade::get_max_concurrency() const {
    return strategy_->get_max_concurrency();
}

bool ParallelismFacade::is_target_available(HardwareTarget target) const {
    return ExecutionStrategyFactory::is_target_available(target);
}

std::vector<HardwareTarget> ParallelismFacade::get_available_targets() const {
    return ExecutionStrategyFactory::get_available_targets();
}

ParallelismFacade::PerformanceMetrics ParallelismFacade::get_performance_metrics() const {
    // For now, return placeholder metrics
    // In a full implementation, this would collect real performance data
    PerformanceMetrics metrics;
    metrics.tasks_executed = 0;
    metrics.avg_execution_time_ms = 0.0;
    metrics.throughput_tasks_per_sec = 0.0;
    metrics.active_workers = get_max_concurrency();
    return metrics;
}

} // namespace diffeq::execution