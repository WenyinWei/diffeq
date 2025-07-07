#pragma once
#include <concepts>
#include <type_traits>
#include <string>

// Time type concept - basic arithmetic types that can represent time
template<typename T>
concept can_be_time = std::is_arithmetic_v<T>;

// State concept - supports vectors, matrices, multi-dimensional tensors, etc.
template<typename T>
concept system_state = requires(T state) {
    typename T::value_type;
    requires std::is_arithmetic_v<typename T::value_type>;
    requires !std::same_as<T, std::string>; // Exclude string types
    { state.size() } -> std::convertible_to<std::size_t>;
    { state.begin() };
    { state.end() };
    { state[0] } -> std::convertible_to<typename T::value_type>;
};