#pragma once

#ifdef USE_XTENSOR
#include <xtensor/xarray.hpp>
#endif

namespace diffeq {
    class XTensorAdapter : public ITensorAdapterFactory {
    public:
        IState* create_state() override {
#ifdef USE_XTENSOR
            return new XTensorState();
#else
            static_assert(false, "XTensor not enabled");
#endif
        }
    };
}