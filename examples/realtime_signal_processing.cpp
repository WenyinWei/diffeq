#include <diffeq.hpp>
#include <interfaces/integration_interface.hpp>
#include <async/async_integrator.hpp>
#include <signal/signal_processor.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>

// Example: Real-time signal-driven ODE integration (finance-style)
void realtime_signal_processing_example() {
    using State = std::vector<double>;
    std::cout << "=== Real-time Signal Processing Example ===\n";

    // Initial state: [asset1, asset2, asset3]
    State state = {100.0, 150.0, 120.0};

    // Create signal-aware integration interface
    auto interface = diffeq::interfaces::make_integration_interface<State, double>();

    // Register a signal that modifies the first asset value (e.g., price update)
    interface->register_signal_influence<double>(
        "price_update",
        diffeq::interfaces::IntegrationInterface<State, double>::InfluenceMode::CONTINUOUS_SHIFT,
        [](const double& price, State& state, double /*t*/) {
            if (!state.empty()) state[0] += price;
        }
    );

    // Register a discrete event (e.g., risk alert)
    interface->register_signal_influence<std::string>(
        "risk_alert",
        diffeq::interfaces::IntegrationInterface<State, double>::InfluenceMode::DISCRETE_EVENT,
        [](const std::string& alert, State& state, double /*t*/) {
            if (alert == "high_volatility") {
                for (auto& v : state) v *= 0.9;
            }
        }
    );

    // Register an output stream for monitoring
    interface->register_output_stream(
        "monitor",
        [](const State& state, double t) {
            std::cout << "t=" << t << ": [";
            for (auto v : state) std::cout << v << ", ";
            std::cout << "]\n";
        },
        std::chrono::milliseconds{100}
    );

    // Define the ODE system (simple growth)
    auto ode = [](double /*t*/, const State& y, State& dydt) {
        dydt.resize(y.size());
        for (size_t i = 0; i < y.size(); ++i) dydt[i] = 0.01 * y[i];
    };

    // Make the ODE signal-aware
    auto signal_ode = interface->make_signal_aware_ode(ode);

    // Create async integrator
    auto integrator = diffeq::async::factory::make_async_dop853<State>(signal_ode);
    integrator->start();

    // Get the signal processor
    auto signal_proc = interface->get_signal_processor();

    // Simulate real-time signal emission and integration
    double t = 0.0, dt = 0.05, t_end = 2.0;
    std::default_random_engine rng;
    std::normal_distribution<double> price_dist(0.0, 0.5);

    while (t < t_end) {
        // Emit a price update signal occasionally
        if (static_cast<int>(t * 10) % 5 == 0) {
            double price_jump = price_dist(rng);
            signal_proc->emit_signal("price_update", price_jump);
        }
        // Emit a risk alert at t=1.0
        if (std::abs(t - 1.0) < 1e-6) {
            signal_proc->emit_signal("risk_alert", std::string("high_volatility"));
        }
        // Integrate one step
        integrator->step(state, dt);
        t += dt;
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    std::cout << "Final state: [";
    for (auto v : state) std::cout << v << ", ";
    std::cout << "]\n";
}

int main() {
    realtime_signal_processing_example();
    return 0;
}
