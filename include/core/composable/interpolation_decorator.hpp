#pragma once

#include "integrator_decorator.hpp"
#include <vector>
#include <map>
#include <algorithm>
#include <functional>
#include <memory>
#include <stdexcept>
#include <cmath>

namespace diffeq::core::composable {

/**
 * @brief Interpolation method enumeration
 */
enum class InterpolationMethod {
    LINEAR,         // Linear interpolation
    CUBIC_SPLINE,   // Cubic spline interpolation
    HERMITE,        // Hermite polynomial interpolation
    AKIMA           // Akima spline (smooth, avoids oscillation)
};

/**
 * @brief Configuration for interpolation and dense output
 */
struct InterpolationConfig {
    InterpolationMethod method{InterpolationMethod::CUBIC_SPLINE};
    size_t max_history_size{10000};
    bool enable_adaptive_sampling{true};
    double adaptive_tolerance{1e-6};
    double min_step_size{1e-12};
    
    // Memory management
    bool enable_compression{false};
    size_t compression_threshold{1000};
    double compression_tolerance{1e-8};
    
    // Extrapolation settings
    bool allow_extrapolation{false};
    double extrapolation_warning_threshold{0.1};  // Warn if extrapolating beyond 10% of range
    
    /**
     * @brief Validate configuration parameters
     * @throws std::invalid_argument if configuration is invalid
     */
    void validate() const {
        if (max_history_size < 2) {
            throw std::invalid_argument("max_history_size must be at least 2 for interpolation");
        }
        
        if (adaptive_tolerance <= 0) {
            throw std::invalid_argument("adaptive_tolerance must be positive");
        }
        
        if (min_step_size <= 0) {
            throw std::invalid_argument("min_step_size must be positive");
        }
        
        if (compression_threshold > max_history_size) {
            throw std::invalid_argument("compression_threshold cannot exceed max_history_size");
        }
    }
};

/**
 * @brief Statistics for interpolation operations
 */
struct InterpolationStats {
    size_t total_interpolations{0};
    size_t history_compressions{0};
    size_t extrapolation_warnings{0};
    size_t out_of_bounds_queries{0};
    double max_interpolation_error{0.0};
    double average_interpolation_time_ns{0.0};
    
    void update_interpolation_time(double time_ns) {
        average_interpolation_time_ns = (average_interpolation_time_ns * total_interpolations + time_ns) / (total_interpolations + 1);
        total_interpolations++;
    }
};

/**
 * @brief Cubic spline interpolator implementation
 */
template<typename T>
class CubicSplineInterpolator {
private:
    std::vector<T> times_;
    std::vector<std::vector<typename T::value_type>> states_;
    std::vector<std::vector<typename T::value_type>> derivatives_;
    bool computed_{false};
    
public:
    void set_data(const std::vector<T>& times, const std::vector<std::vector<typename T::value_type>>& states) {
        times_ = times;
        states_ = states;
        computed_ = false;
        compute_derivatives();
    }
    
    std::vector<typename T::value_type> interpolate(T t) {
        if (!computed_) {
            throw std::runtime_error("Spline not computed");
        }
        
        if (times_.empty()) {
            throw std::runtime_error("No data for interpolation");
        }
        
        // Find the interval containing t
        auto it = std::lower_bound(times_.begin(), times_.end(), t);
        
        if (it == times_.begin()) {
            return states_[0];  // Extrapolate to first point
        }
        
        if (it == times_.end()) {
            return states_.back();  // Extrapolate to last point
        }
        
        size_t idx = std::distance(times_.begin(), it) - 1;
        T h = times_[idx + 1] - times_[idx];
        T a = (times_[idx + 1] - t) / h;
        T b = (t - times_[idx]) / h;
        
        std::vector<typename T::value_type> result(states_[idx].size());
        
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = a * states_[idx][i] + b * states_[idx + 1][i] +
                       ((a * a * a - a) * derivatives_[idx][i] + (b * b * b - b) * derivatives_[idx + 1][i]) * (h * h) / 6.0;
        }
        
        return result;
    }
    
private:
    void compute_derivatives() {
        size_t n = times_.size();
        if (n < 2) return;
        
        derivatives_.resize(n);
        for (size_t i = 0; i < n; ++i) {
            derivatives_[i].resize(states_[i].size());
        }
        
        if (n == 2) {
            // Linear case
            for (size_t j = 0; j < states_[0].size(); ++j) {
                derivatives_[0][j] = derivatives_[1][j] = 0.0;
            }
            computed_ = true;
            return;
        }
        
        // Tridiagonal system solution for cubic spline
        std::vector<typename T::value_type> a(n), b(n), c(n);
        
        for (size_t j = 0; j < states_[0].size(); ++j) {
            std::vector<typename T::value_type> d(n);
            
            // Set up tridiagonal system
            for (size_t i = 1; i < n - 1; ++i) {
                T h1 = times_[i] - times_[i - 1];
                T h2 = times_[i + 1] - times_[i];
                
                a[i] = h1;
                b[i] = 2.0 * (h1 + h2);
                c[i] = h2;
                d[i] = 6.0 * ((states_[i + 1][j] - states_[i][j]) / h2 - (states_[i][j] - states_[i - 1][j]) / h1);
            }
            
            // Natural boundary conditions
            b[0] = b[n - 1] = 1.0;
            c[0] = a[n - 1] = 0.0;
            d[0] = d[n - 1] = 0.0;
            
            // Solve tridiagonal system
            solve_tridiagonal(a, b, c, d);
            
            for (size_t i = 0; i < n; ++i) {
                derivatives_[i][j] = d[i];
            }
        }
        
        computed_ = true;
    }
    
    void solve_tridiagonal(std::vector<typename T::value_type>& a, 
                          std::vector<typename T::value_type>& b, 
                          std::vector<typename T::value_type>& c, 
                          std::vector<typename T::value_type>& d) {
        size_t n = b.size();
        
        // Forward elimination
        for (size_t i = 1; i < n; ++i) {
            typename T::value_type m = a[i] / b[i - 1];
            b[i] = b[i] - m * c[i - 1];
            d[i] = d[i] - m * d[i - 1];
        }
        
        // Back substitution
        d[n - 1] = d[n - 1] / b[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            d[i] = (d[i] - c[i] * d[i + 1]) / b[i];
        }
    }
};

/**
 * @brief Interpolation decorator - adds dense output capabilities to any integrator
 * 
 * This decorator provides comprehensive interpolation capabilities with the following features:
 * - Dense output with multiple interpolation methods
 * - Adaptive sampling and history management
 * - Memory-efficient compression
 * - Query interface for arbitrary time points
 * 
 * Key Design Principles:
 * - Single Responsibility: ONLY handles interpolation and dense output
 * - Efficient: Minimal memory overhead with compression
 * - Flexible: Multiple interpolation methods
 * - Robust: Handles edge cases and extrapolation
 */
template<system_state S>
class InterpolationDecorator : public IntegratorDecorator<S> {
private:
    InterpolationConfig config_;
    std::map<typename IntegratorDecorator<S>::time_type, S> state_history_;
    std::unique_ptr<CubicSplineInterpolator<typename IntegratorDecorator<S>::time_type>> spline_interpolator_;
    InterpolationStats stats_;
    mutable std::mutex history_mutex_;
    typename IntegratorDecorator<S>::time_type last_query_time_{};
    bool history_compressed_{false};

public:
    /**
     * @brief Construct interpolation decorator
     * @param integrator The integrator to wrap
     * @param config Interpolation configuration (validated on construction)
     * @throws std::invalid_argument if config is invalid
     */
    explicit InterpolationDecorator(std::unique_ptr<AbstractIntegrator<S>> integrator,
                                   InterpolationConfig config = {})
        : IntegratorDecorator<S>(std::move(integrator))
        , config_(std::move(config))
        , spline_interpolator_(std::make_unique<CubicSplineInterpolator<typename IntegratorDecorator<S>::time_type>>()) {
        
        config_.validate();
    }

    /**
     * @brief Override step to record state history
     */
    void step(typename IntegratorDecorator<S>::state_type& state, typename IntegratorDecorator<S>::time_type dt) override {
        this->wrapped_integrator_->step(state, dt);
        record_state(state, this->current_time());
    }

    /**
     * @brief Override integrate to maintain history during integration
     */
    void integrate(typename IntegratorDecorator<S>::state_type& state, typename IntegratorDecorator<S>::time_type dt, 
                   typename IntegratorDecorator<S>::time_type end_time) override {
        // Record initial state
        record_state(state, this->current_time());
        
        // Integrate with history recording
        this->wrapped_integrator_->integrate(state, dt, end_time);
        
        // Record final state
        record_state(state, this->current_time());
        
        // Compress history if needed
        if (config_.enable_compression && state_history_.size() > config_.compression_threshold) {
            compress_history();
        }
    }

    /**
     * @brief Get interpolated state at arbitrary time
     * @param t Time for interpolation
     * @return Interpolated state
     * @throws std::runtime_error if interpolation fails
     */
    S interpolate_at(typename IntegratorDecorator<S>::time_type t) {
        std::lock_guard<std::mutex> lock(history_mutex_);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (state_history_.empty()) {
            throw std::runtime_error("No history available for interpolation");
        }
        
        // Check bounds
        auto bounds = get_time_bounds();
        if (t < bounds.first || t > bounds.second) {
            if (!config_.allow_extrapolation) {
                stats_.out_of_bounds_queries++;
                throw std::runtime_error("Time " + std::to_string(t) + " is outside interpolation bounds [" + 
                                       std::to_string(bounds.first) + ", " + std::to_string(bounds.second) + "]");
            }
            
            // Check extrapolation warning threshold
            typename IntegratorDecorator<S>::time_type range = bounds.second - bounds.first;
            if (std::abs(t - bounds.first) > config_.extrapolation_warning_threshold * range ||
                std::abs(t - bounds.second) > config_.extrapolation_warning_threshold * range) {
                stats_.extrapolation_warnings++;
            }
        }
        
        S result = perform_interpolation(t);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        stats_.update_interpolation_time(duration_ns);
        
        last_query_time_ = t;
        return result;
    }

    /**
     * @brief Get interpolated states at multiple time points
     * @param time_points Vector of time points
     * @return Vector of interpolated states
     */
    std::vector<S> interpolate_at_multiple(const std::vector<typename IntegratorDecorator<S>::time_type>& time_points) {
        std::vector<S> results;
        results.reserve(time_points.size());
        
        for (typename IntegratorDecorator<S>::time_type t : time_points) {
            results.push_back(interpolate_at(t));
        }
        
        return results;
    }

    /**
     * @brief Get dense output over time interval
     * @param start_time Start time
     * @param end_time End time
     * @param num_points Number of interpolation points
     * @return Pair of time vector and state vector
     */
    std::pair<std::vector<typename IntegratorDecorator<S>::time_type>, std::vector<S>> get_dense_output(
        typename IntegratorDecorator<S>::time_type start_time, 
        typename IntegratorDecorator<S>::time_type end_time, 
        size_t num_points) {
        if (num_points < 2) {
            throw std::invalid_argument("num_points must be at least 2");
        }
        
        std::vector<typename IntegratorDecorator<S>::time_type> times;
        std::vector<S> states;
        
        typename IntegratorDecorator<S>::time_type dt = (end_time - start_time) / (num_points - 1);
        
        for (size_t i = 0; i < num_points; ++i) {
            typename IntegratorDecorator<S>::time_type t = start_time + i * dt;
            times.push_back(t);
            states.push_back(interpolate_at(t));
        }
        
        return {std::move(times), std::move(states)};
    }

    /**
     * @brief Get current interpolation statistics
     */
    const InterpolationStats& get_statistics() const {
        return stats_;
    }

    /**
     * @brief Reset interpolation statistics
     */
    void reset_statistics() {
        stats_ = InterpolationStats{};
    }

    /**
     * @brief Get time bounds of available history
     * @return Pair of (min_time, max_time)
     */
    std::pair<typename IntegratorDecorator<S>::time_type, typename IntegratorDecorator<S>::time_type> get_time_bounds() const {
        std::lock_guard<std::mutex> lock(history_mutex_);
        if (state_history_.empty()) {
            return {typename IntegratorDecorator<S>::time_type{}, typename IntegratorDecorator<S>::time_type{}};
        }
        return {state_history_.begin()->first, state_history_.rbegin()->first};
    }

    /**
     * @brief Clear all history
     */
    void clear_history() {
        std::lock_guard<std::mutex> lock(history_mutex_);
        state_history_.clear();
        history_compressed_ = false;
    }

    /**
     * @brief Get number of stored history points
     */
    size_t get_history_size() const {
        std::lock_guard<std::mutex> lock(history_mutex_);
        return state_history_.size();
    }

    /**
     * @brief Access and modify interpolation configuration
     */
    InterpolationConfig& config() { return config_; }
    const InterpolationConfig& config() const { return config_; }

private:
    /**
     * @brief Record state at given time
     */
    void record_state(const S& state, typename IntegratorDecorator<S>::time_type time) {
        std::lock_guard<std::mutex> lock(history_mutex_);
        
        // Check if we need to make room
        if (state_history_.size() >= config_.max_history_size) {
            // Remove oldest entry
            state_history_.erase(state_history_.begin());
        }
        
        state_history_[time] = state;
    }

    /**
     * @brief Perform interpolation using configured method
     */
    S perform_interpolation(typename IntegratorDecorator<S>::time_type t) {
        switch (config_.method) {
            case InterpolationMethod::LINEAR:
                return linear_interpolation(t);
            case InterpolationMethod::CUBIC_SPLINE:
                return cubic_spline_interpolation(t);
            case InterpolationMethod::HERMITE:
                return hermite_interpolation(t);
            case InterpolationMethod::AKIMA:
                return akima_interpolation(t);
            default:
                throw std::runtime_error("Unknown interpolation method");
        }
    }

    /**
     * @brief Linear interpolation
     */
    S linear_interpolation(typename IntegratorDecorator<S>::time_type t) {
        auto it = state_history_.lower_bound(t);
        
        if (it == state_history_.begin()) {
            return it->second;
        }
        
        if (it == state_history_.end()) {
            return state_history_.rbegin()->second;
        }
        
        auto prev_it = std::prev(it);
        
        typename IntegratorDecorator<S>::time_type t1 = prev_it->first;
        typename IntegratorDecorator<S>::time_type t2 = it->first;
        const S& s1 = prev_it->second;
        const S& s2 = it->second;
        
        typename IntegratorDecorator<S>::time_type alpha = (t - t1) / (t2 - t1);
        
        S result = s1;
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = (1 - alpha) * s1[i] + alpha * s2[i];
        }
        
        return result;
    }

    /**
     * @brief Cubic spline interpolation
     */
    S cubic_spline_interpolation(typename IntegratorDecorator<S>::time_type t) {
        // Prepare data for spline interpolator
        std::vector<typename IntegratorDecorator<S>::time_type> times;
        std::vector<std::vector<typename S::value_type>> states;
        
        times.reserve(state_history_.size());
        states.reserve(state_history_.size());
        
        for (const auto& [time, state] : state_history_) {
            times.push_back(time);
            std::vector<typename S::value_type> state_vec(state.begin(), state.end());
            states.push_back(std::move(state_vec));
        }
        
        spline_interpolator_->set_data(times, states);
        auto result_vec = spline_interpolator_->interpolate(t);
        
        // Convert back to state type
        S result;
        if constexpr (std::is_same_v<S, std::vector<typename S::value_type>>) {
            result = result_vec;
        } else {
            std::copy(result_vec.begin(), result_vec.end(), result.begin());
        }
        
        return result;
    }

    /**
     * @brief Hermite interpolation (placeholder - can be extended)
     */
    S hermite_interpolation(typename IntegratorDecorator<S>::time_type t) {
        // For now, fall back to cubic spline
        return cubic_spline_interpolation(t);
    }

    /**
     * @brief Akima interpolation (placeholder - can be extended)
     */
    S akima_interpolation(typename IntegratorDecorator<S>::time_type t) {
        // For now, fall back to cubic spline
        return cubic_spline_interpolation(t);
    }

    /**
     * @brief Compress history by removing redundant points
     */
    void compress_history() {
        if (state_history_.size() <= config_.compression_threshold) {
            return;
        }
        
        // Simple compression: remove every other point if error is small
        auto it = state_history_.begin();
        while (it != state_history_.end() && state_history_.size() > config_.compression_threshold / 2) {
            auto next_it = std::next(it);
            if (next_it != state_history_.end()) {
                auto next_next_it = std::next(next_it);
                if (next_next_it != state_history_.end()) {
                    // Check if middle point can be removed
                    if (is_point_redundant(*it, *next_it, *next_next_it)) {
                        it = state_history_.erase(next_it);
                        continue;
                    }
                }
            }
            ++it;
        }
        
        stats_.history_compressions++;
        history_compressed_ = true;
    }

    /**
     * @brief Check if a point is redundant for compression
     */
    bool is_point_redundant(const std::pair<typename IntegratorDecorator<S>::time_type, S>& p1, 
                           const std::pair<typename IntegratorDecorator<S>::time_type, S>& p2, 
                           const std::pair<typename IntegratorDecorator<S>::time_type, S>& p3) {
        // Simple linear interpolation error check
        typename IntegratorDecorator<S>::time_type alpha = (p2.first - p1.first) / (p3.first - p1.first);
        
        for (size_t i = 0; i < p2.second.size(); ++i) {
            double interpolated = (1 - alpha) * p1.second[i] + alpha * p3.second[i];
            double error = std::abs(interpolated - p2.second[i]);
            if (error > config_.compression_tolerance) {
                return false;
            }
        }
        
        return true;
    }
};

} // namespace diffeq::core::composable 