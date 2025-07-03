#pragma once

#ifdef USE_XTENSOR
#include <xtensor/xarray.hpp>
using XTensor = xt::xarray<double>;
#endif

#ifdef USE_EIGEN
#include <Eigen/Dense>
using EigenMatrix = Eigen::MatrixXd;
#endif

namespace diffeq {
    class IState {
    public:
        virtual ~IState() = default;
        virtual void* data_ptr() = 0;
        
        // 多态数据访问接口
        template<typename T>
        T get_data() {
            #ifdef USE_XTENSOR
            return static_cast<T*>(data_ptr());
            #elif defined(USE_EIGEN)
            return *static_cast<T*>(data_ptr());
            #else
            static_assert(false, "No tensor backend enabled");
            #endif
        }
    };
}