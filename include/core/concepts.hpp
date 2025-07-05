#pragma once
#include <concepts>
#include <type_traits>
#include <iterator>
#include <string>

// Time type concept - supports both integer and floating point time
template<typename T>
concept can_be_time = std::is_arithmetic_v<T>;

// 状态概念 - 支持向量、矩阵、多维张量等类型
template<typename T>
concept system_state = requires(T state) {
    typename T::value_type;
    requires std::is_arithmetic_v<typename T::value_type>;
    requires !std::same_as<T, std::string>; // Exclude string types
    requires requires {
        { state.size() } -> std::convertible_to<std::size_t>;
        { state.begin() } -> std::random_access_iterator;
        { state.end() } -> std::random_access_iterator;
    };
};