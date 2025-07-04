#include <examples/sde_usage.hpp>

int main() {
    try {
        // Run comprehensive demonstration of SDE solvers
        diffeq::examples::sde::run_comprehensive_sde_examples();
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
