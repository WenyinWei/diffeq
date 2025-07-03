#include "diffeq_core.h"
#include "plugin_system.h"

int main() {
    // 使用纯C++状态实现
    class MyState : public IState {
        double data[3];
    public:
        void* data_ptr() override { return data; }
        size_t size() const override { return 3; }
    };
    
    auto factory = plugin_system::get_factory("default_state");
    IState* state = factory->create_state();
}