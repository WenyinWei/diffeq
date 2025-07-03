#pragma once

#include <memory>

namespace diffeq {
    class IState {
    public:
        virtual ~IState() = default;
        virtual void* data_ptr() = 0;
        
        template<typename T>
        T get_data() {
            return *static_cast<T*>(data_ptr());
        }
    };
}