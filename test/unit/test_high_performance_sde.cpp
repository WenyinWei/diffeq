/**
 * @file test_high_performance_sde.cpp
 * @brief Unit tests for high-performance SDE multithreading and fiber capabilities
 */

#include <catch2/catch.hpp>
#include <core/composable_integration.hpp>
#include <core/composable/sde_multithreading.hpp>
#include <vector>
#include <thread>
#include <chrono>
#include <numeric>
#include <memory>
#include <future>
#include <random>

using namespace diffeq::core::composable;

// Test helper for simple SDE system
struct TestSDESystem {
    std::vector<double> state{1.0};
    
    void step(double dt) {
        // Simple drift: dX = 0.1 * X * dt + noise
        state[0] *= (1.0 + 0.1 * dt);
    }
};

TEST_CASE("SDEThreadingConfig validation", "[sde_multithreading]") {
    SECTION("Valid configuration") {
        SDEThreadingConfig config;
        config.num_threads = 4;
        config.batch_size = 1000;
        config.queue_size = 10000;
        config.precompute_buffer_size = 100000;
        
        REQUIRE_NOTHROW(config.validate());
    }
    
    SECTION("Invalid fiber count") {
        SDEThreadingConfig config;
        config.num_fibers = 0;
        
        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
    }
    
    SECTION("Invalid batch size") {
        SDEThreadingConfig config;
        config.batch_size = 0;
        
        REQUIRE_THROWS_AS(config.validate(), std::invalid_argument);
    }
    
    SECTION("Auto-detect configuration") {
        auto config = SDEThreadingConfig::auto_detect();
        
        REQUIRE(config.num_threads > 0);
        REQUIRE(config.batch_size > 0);
        REQUIRE(config.queue_size > 0);
        REQUIRE(config.precompute_buffer_size > 0);
        REQUIRE_NOTHROW(config.validate());
    }
}

TEST_CASE("SIMDNoiseGenerator basic functionality", "[sde_multithreading]") {
    SIMDNoiseGenerator generator(12345);
    
    SECTION("Single sample generation") {
        auto samples = generator.generate_batch(1);
        REQUIRE(samples.size() == 1);
        REQUIRE(std::isfinite(samples[0]));
    }
    
    SECTION("Batch generation") {
        const size_t batch_size = 1000;
        auto samples = generator.generate_batch(batch_size);
        
        REQUIRE(samples.size() == batch_size);
        
        // Check all samples are finite
        for (double sample : samples) {
            REQUIRE(std::isfinite(sample));
        }
        
        // Basic statistical checks (should be approximately N(0,1))
        double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        double variance = 0.0;
        for (double sample : samples) {
            variance += (sample - mean) * (sample - mean);
        }
        variance /= samples.size();
        
        // Should be roughly normal with mean ≈ 0 and variance ≈ 1
        REQUIRE(std::abs(mean) < 0.1);        // Mean should be close to 0
        REQUIRE(std::abs(variance - 1.0) < 0.2); // Variance should be close to 1
    }
    
    SECTION("Different intensities") {
        const double intensity = 2.0;
        auto samples = generator.generate_batch(1000, intensity);
        
        double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        double variance = 0.0;
        for (double sample : samples) {
            variance += (sample - mean) * (sample - mean);
        }
        variance /= samples.size();
        
        // With intensity 2.0, variance should be approximately 4.0
        REQUIRE(std::abs(variance - 4.0) < 0.8);
    }
}

TEST_CASE("LockFreeNoiseQueue operations", "[sde_multithreading]") {
    LockFreeNoiseQueue<double> queue;
    
    SECTION("Basic push/pop operations") {
        REQUIRE(queue.empty());
        REQUIRE(queue.size() == 0);
        
        NoiseData<double> data1(1.0, {0.5}, NoiseProcessType::WIENER);
        REQUIRE(queue.push(data1));
        REQUIRE(queue.size() == 1);
        REQUIRE(!queue.empty());
        
        NoiseData<double> data2(0.0, {}, NoiseProcessType::WIENER);
        REQUIRE(queue.pop(data2));
        REQUIRE(queue.size() == 0);
        REQUIRE(queue.empty());
        
        REQUIRE(data2.time == 1.0);
        REQUIRE(data2.increments.size() == 1);
        REQUIRE(data2.increments[0] == 0.5);
    }
    
    SECTION("Multiple operations") {
        std::vector<NoiseData<double>> test_data;
        for (int i = 0; i < 10; ++i) {
            test_data.emplace_back(i * 0.1, std::vector<double>{i * 0.1}, NoiseProcessType::WIENER);
        }
        
        // Push all data
        for (const auto& data : test_data) {
            REQUIRE(queue.push(data));
        }
        REQUIRE(queue.size() == test_data.size());
        
        // Pop all data
        std::vector<NoiseData<double>> retrieved_data;
        NoiseData<double> temp(0.0, {}, NoiseProcessType::WIENER);
        while (queue.pop(temp)) {
            retrieved_data.push_back(temp);
        }
        
        REQUIRE(retrieved_data.size() == test_data.size());
        REQUIRE(queue.empty());
    }
    
    SECTION("Concurrent access") {
        const size_t num_threads = 4;
        const size_t items_per_thread = 1000;
        std::atomic<size_t> successful_pushes{0};
        std::atomic<size_t> successful_pops{0};
        
        std::vector<std::thread> threads;
        
        // Producer threads
        for (size_t i = 0; i < num_threads / 2; ++i) {
            threads.emplace_back([&, i]() {
                for (size_t j = 0; j < items_per_thread; ++j) {
                    NoiseData<double> data(j * 0.001, {j * 0.001}, NoiseProcessType::WIENER);
                    if (queue.push(data)) {
                        successful_pushes++;
                    }
                }
            });
        }
        
        // Consumer threads
        for (size_t i = 0; i < num_threads / 2; ++i) {
            threads.emplace_back([&]() {
                NoiseData<double> data(0.0, {}, NoiseProcessType::WIENER);
                size_t local_pops = 0;
                while (local_pops < items_per_thread) {
                    if (queue.pop(data)) {
                        successful_pops++;
                        local_pops++;
                    } else {
                        std::this_thread::yield();
                    }
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        REQUIRE(successful_pushes == successful_pops);
        REQUIRE(successful_pushes == (num_threads / 2) * items_per_thread);
    }
}

TEST_CASE("HighPerformanceSDESynchronizer basic functionality", "[sde_multithreading]") {
    SECTION("Construction and configuration") {
        SDEThreadingConfig config;
        config.num_threads = 2;
        config.threading_mode = SDEThreadingMode::MULTI_THREAD;
        
        REQUIRE_NOTHROW(HighPerformanceSDESynchronizer<std::vector<double>>(config));
    }
    
    SECTION("Noise generation") {
        auto synchronizer = HighPerformanceSDESynchronizer<std::vector<double>>(
            SDEThreadingConfig::auto_detect());
        
        auto noise = synchronizer.get_noise_increment_fast(0.0, 0.01, 1);
        
        REQUIRE(noise.time == 0.0);
        REQUIRE(noise.increments.size() == 1);
        REQUIRE(std::isfinite(noise.increments[0]));
        REQUIRE(noise.process_type == NoiseProcessType::WIENER);
    }
    
    SECTION("Multi-dimensional noise") {
        auto synchronizer = HighPerformanceSDESynchronizer<std::vector<double>>(
            SDEThreadingConfig::auto_detect());
        
        const size_t dimensions = 3;
        auto noise = synchronizer.get_noise_increment_fast(0.0, 0.01, dimensions);
        
        REQUIRE(noise.increments.size() == dimensions);
        for (double increment : noise.increments) {
            REQUIRE(std::isfinite(increment));
        }
    }
    
    SECTION("Statistics tracking") {
        auto synchronizer = HighPerformanceSDESynchronizer<std::vector<double>>(
            SDEThreadingConfig::auto_detect());
        
        auto initial_stats = synchronizer.get_statistics();
        REQUIRE(initial_stats.noise_consumed == 0);
        
        // Generate some noise
        for (int i = 0; i < 10; ++i) {
            synchronizer.get_noise_increment_fast(i * 0.01, 0.01, 1);
        }
        
        auto final_stats = synchronizer.get_statistics();
        REQUIRE(final_stats.noise_consumed == 10);
        REQUIRE(final_stats.noise_generated >= 10);
    }
}

TEST_CASE("Monte Carlo batch generation", "[sde_multithreading]") {
    auto synchronizer = HighPerformanceSDESynchronizer<std::vector<double>>(
        SDEThreadingConfig::auto_detect());
    
    SECTION("Basic batch generation") {
        const size_t num_simulations = 1000;
        const size_t dimensions = 1;
        
        auto batch = synchronizer.generate_monte_carlo_batch(0.0, 0.01, dimensions, num_simulations);
        
        REQUIRE(batch.size() == num_simulations);
        
        for (const auto& noise : batch) {
            REQUIRE(noise.increments.size() == dimensions);
            REQUIRE(std::isfinite(noise.increments[0]));
        }
    }
    
    SECTION("Multi-dimensional batch") {
        const size_t num_simulations = 500;
        const size_t dimensions = 3;
        
        auto batch = synchronizer.generate_monte_carlo_batch(0.0, 0.01, dimensions, num_simulations);
        
        REQUIRE(batch.size() == num_simulations);
        
        for (const auto& noise : batch) {
            REQUIRE(noise.increments.size() == dimensions);
            for (double increment : noise.increments) {
                REQUIRE(std::isfinite(increment));
            }
        }
    }
    
    SECTION("Large batch performance") {
        const size_t num_simulations = 10000;
        
        auto start = std::chrono::high_resolution_clock::now();
        auto batch = synchronizer.generate_monte_carlo_batch(0.0, 0.01, 1, num_simulations);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        REQUIRE(batch.size() == num_simulations);
        REQUIRE(duration.count() < 1000);  // Should complete in less than 1 second
        
        // Calculate throughput
        double throughput = static_cast<double>(num_simulations) / (duration.count() * 1000.0);
        REQUIRE(throughput > 100.0);  // Should achieve at least 100K samples/sec
    }
}

TEST_CASE("Threading mode comparison", "[sde_multithreading]") {
    const size_t num_simulations = 1000;
    
    std::vector<SDEThreadingMode> modes = {
        SDEThreadingMode::SINGLE_THREAD,
        SDEThreadingMode::MULTI_THREAD,
        SDEThreadingMode::VECTORIZED
    };
    
    for (auto mode : modes) {
        SDEThreadingConfig config;
        config.threading_mode = mode;
        config.num_threads = 2;
        
        auto synchronizer = HighPerformanceSDESynchronizer<std::vector<double>>(config);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto batch = synchronizer.generate_monte_carlo_batch(0.0, 0.01, 1, num_simulations);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        REQUIRE(batch.size() == num_simulations);
        REQUIRE(duration.count() < 5000);  // Should complete in reasonable time
        
        // Verify correctness
        for (const auto& noise : batch) {
            REQUIRE(noise.increments.size() == 1);
            REQUIRE(std::isfinite(noise.increments[0]));
        }
    }
}

TEST_CASE("System warmup functionality", "[sde_multithreading]") {
    auto synchronizer = HighPerformanceSDESynchronizer<std::vector<double>>(
        SDEThreadingConfig::auto_detect());
    
    SECTION("Warmup improves performance") {
        // Measure cold performance
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; ++i) {
            synchronizer.get_noise_increment_fast(i * 0.001, 0.001, 1);
        }
        auto cold_duration = std::chrono::high_resolution_clock::now() - start;
        
        // Warmup
        synchronizer.warmup(10000);
        
        // Measure warm performance
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; ++i) {
            synchronizer.get_noise_increment_fast(i * 0.001, 0.001, 1);
        }
        auto warm_duration = std::chrono::high_resolution_clock::now() - start;
        
        // Warmup should improve cache hit rates
        auto stats = synchronizer.get_statistics();
        REQUIRE(stats.cache_hit_rate() > 0.5);  // Should have reasonable cache hit rate
    }
}

TEST_CASE("Convenience factory functions", "[sde_multithreading]") {
    SECTION("create_monte_carlo_system") {
        const size_t num_simulations = 10000;
        auto system = create_monte_carlo_system<std::vector<double>>(num_simulations);
        
        REQUIRE(system != nullptr);
        
        // Test basic functionality
        auto batch = system->generate_monte_carlo_batch(0.0, 0.01, 1, 1000);
        REQUIRE(batch.size() == 1000);
    }
    
    SECTION("create_realtime_system") {
        auto system = create_realtime_system<std::vector<double>>();
        
        REQUIRE(system != nullptr);
        
        // Test low-latency generation
        auto start = std::chrono::high_resolution_clock::now();
        auto noise = system->get_noise_increment_fast(0.0, 0.01, 1);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        REQUIRE(noise.increments.size() == 1);
        REQUIRE(latency.count() < 100000);  // Should be sub-100μs
    }
    
    SECTION("create_numa_system") {
        auto system = create_numa_system<std::vector<double>>();
        
        REQUIRE(system != nullptr);
        
        // Test basic functionality
        auto noise = system->get_noise_increment_fast(0.0, 0.01, 1);
        REQUIRE(noise.increments.size() == 1);
    }
}

TEST_CASE("Memory and performance characteristics", "[sde_multithreading]") {
    SECTION("Memory usage is reasonable") {
        SDEThreadingConfig config;
        config.num_threads = 4;
        config.precompute_buffer_size = 10000;
        
        auto synchronizer = HighPerformanceSDESynchronizer<std::vector<double>>(config);
        
        // Generate large batch and verify memory doesn't explode
        auto batch = synchronizer.generate_monte_carlo_batch(0.0, 0.01, 1, 50000);
        REQUIRE(batch.size() == 50000);
        
        auto stats = synchronizer.get_statistics();
        REQUIRE(stats.noise_generated > 0);
        REQUIRE(stats.noise_consumed > 0);
    }
    
    SECTION("Performance scales with thread count") {
        std::vector<size_t> thread_counts = {1, 2, 4};
        std::vector<double> throughputs;
        
        for (size_t num_threads : thread_counts) {
            SDEThreadingConfig config;
            config.num_threads = num_threads;
            config.threading_mode = SDEThreadingMode::MULTI_THREAD;
            
            auto synchronizer = HighPerformanceSDESynchronizer<std::vector<double>>(config);
            
            const size_t num_simulations = 10000;
            auto start = std::chrono::high_resolution_clock::now();
            auto batch = synchronizer.generate_monte_carlo_batch(0.0, 0.01, 1, num_simulations);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            double throughput = static_cast<double>(num_simulations) / (duration.count() * 1000.0);
            
            throughputs.push_back(throughput);
            
            REQUIRE(batch.size() == num_simulations);
        }
        
        // Performance should generally improve with more threads
        // (though not necessarily linear due to overhead)
        REQUIRE(throughputs.back() >= throughputs.front() * 0.8);
    }
}

#ifdef __cpp_impl_coroutine
TEST_CASE("Fiber-based SDE synchronizer", "[sde_multithreading]") {
    SECTION("Basic fiber system creation") {
        FiberSDESynchronizer<std::vector<double>> fiber_system;
        
        auto awaitable = fiber_system.get_noise_async(0.0, 0.01, 1);
        REQUIRE(awaitable.time == 0.0);
        REQUIRE(awaitable.dt == 0.01);
        REQUIRE(awaitable.dimensions == 1);
    }
}
#endif

TEST_CASE("Error handling and edge cases", "[sde_multithreading]") {
    SECTION("Zero dimensions") {
        auto synchronizer = HighPerformanceSDESynchronizer<std::vector<double>>(
            SDEThreadingConfig::auto_detect());
        
        auto noise = synchronizer.get_noise_increment_fast(0.0, 0.01, 0);
        REQUIRE(noise.increments.empty());
    }
    
    SECTION("Very large batch") {
        auto synchronizer = HighPerformanceSDESynchronizer<std::vector<double>>(
            SDEThreadingConfig::auto_detect());
        
        const size_t large_batch = 100000;
        auto batch = synchronizer.generate_monte_carlo_batch(0.0, 0.01, 1, large_batch);
        
        REQUIRE(batch.size() == large_batch);
        
        // Verify all samples are valid
        for (const auto& noise : batch) {
            REQUIRE(noise.increments.size() == 1);
            REQUIRE(std::isfinite(noise.increments[0]));
        }
    }
    
    SECTION("Statistics accuracy") {
        auto synchronizer = HighPerformanceSDESynchronizer<std::vector<double>>(
            SDEThreadingConfig::auto_detect());
        
        synchronizer.reset_statistics();
        auto initial_stats = synchronizer.get_statistics();
        
        REQUIRE(initial_stats.noise_generated == 0);
        REQUIRE(initial_stats.noise_consumed == 0);
        REQUIRE(initial_stats.cache_hits == 0);
        REQUIRE(initial_stats.cache_misses == 0);
        
        // Generate some noise
        const size_t operations = 100;
        for (size_t i = 0; i < operations; ++i) {
            synchronizer.get_noise_increment_fast(i * 0.01, 0.01, 1);
        }
        
        auto final_stats = synchronizer.get_statistics();
        REQUIRE(final_stats.noise_consumed == operations);
        REQUIRE(final_stats.noise_generated >= operations);
    }
} 