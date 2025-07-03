#include <pybind11/pybind11.h>

#ifdef USE_XTENSOR
#include <xtensor-python/pytensor.hpp>
#endif

PYBIND11_MODULE(diffeq_py, m) {
    #ifdef USE_XTENSOR
    py::class_<XTensor>(m, "XTensor")
        .def("__getitem__", [](XTensor& arr, py::tuple idx) {
            return arr(idx[0](@ref).cast<int>(), idx[1](@ref).cast<int>());
        });
    #endif
    
    #ifdef USE_EIGEN
    py::class_<EigenMatrix>(m, "EigenMatrix")
        .def("__call__", [](EigenMatrix& mat, int i, int j) {
            return mat(i, j);
        });
    #endif
}