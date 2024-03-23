/**
 * @file meta_types.h
 * @brief Header file containing unsupported types and user-defined compile-time checks
 *
 * @author Harshavardhan Karnati
 * @date 06/03/2024
 */

#ifndef TYPES_H
#define TYPES_H

#include <type_traits>
#include <functional>
#include <string>
#include <concepts>

/**
 * @brief Namespace containing utility functions for handling type traits and operations.
 *
 * The meta_types namespace provides various utility functions for handling type traits and operations,
 * including checking function invocability, manipulating tuples, and determining type equality.
 */
namespace meta_types {
    /**
     * @brief Removes all qualifiers from a given type.
     *
     * This template alias removes all qualifiers (const, volatile, and reference) from a given type.
     */
    template <class type>
            using remove_all_qual = std::remove_pointer_t<std::remove_cvref_t<type>>;

    /**
     * @brief Checks if a given type can be constructed into a std::string.
     *
     * This trait checks if a given type can be constructed into a std::string using std::is_constructible.
     */
    template <class type>
            static constexpr bool is_string_v = std::is_constructible_v<std::string, std::decay_t<type>>;

    /**
     * @brief Checks if a function of a specified type can be invoked with given argument types.
     *
     * This trait checks if a function of a specified type can be invoked with the given argument types.
     */
    template <class return_type, class func_type, class... args_type>
            static constexpr bool check_func_v = std::is_invocable_r_v<return_type, remove_all_qual<func_type>, args_type...>;

    /**
     * @brief Provides the type of the ith argument in a tuple.
     *
     * This template alias provides the type of the ith argument in a tuple.
     */
    template <std::size_t i, class tuple_type>
            using tuple_args_type_at = meta_types::remove_all_qual<std::tuple_element_t<i, meta_types::remove_all_qual<tuple_type>>>;

    /**
     * @brief Creates a functional type from a given function type and argument types.
     *
     * This template alias creates a functional type from a given function type and argument types.
     */
    template <class funcType, class... argsType>
            using create_functional_type = std::function<meta_types::return_type_t<funcType, argsType...>(argsType...)> ;

    /**
     * @brief Checks if all types in two tuples are the same.
     *
     * This struct template checks if all types in two tuples are the same.
     */
    template <class... types>
    struct are_same : std::true_type{};

    /**
     * @brief Checks if two tuples have the same types.
     *
     * This struct template checks if two tuples have the same types using recursive type comparison.
     */
    template <class T1, class... first_remaining_type, class T2, class... second_remaining_type>
    struct are_same<std::tuple<T1, first_remaining_type...>, std::tuple<T2, second_remaining_type...>> :
            std::conditional_t<std::is_same_v<T1, T2>, are_same<std::tuple<first_remaining_type...>, std::tuple<second_remaining_type...>>, std::false_type>{};

    /**
     * @brief Checks if two tuple types are the same after removing all qualifiers.
     *
     * This trait checks if two tuple types are the same after removing all qualifiers.
     */
    template <class tupleType1_, class tupleType2_>
        static constexpr bool are_tuples_same_v = std::is_same<meta_types::remove_all_qual<tupleType1_>, meta_types::remove_all_qual<tupleType2_>>::value_type;

    // (** removed unnecessary types and checks...)
}


#endif //TYPES_H
