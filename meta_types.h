//
// Created by Harshavardhan Karnati on 06/03/2024.
//

#ifndef TYPES_H
#define TYPES_H

#include <type_traits>
#include <functional>
#include <string>
#include <concepts>

namespace meta_types {
    template <class type>
            using remove_all_qual = std::remove_pointer_t<std::remove_cvref_t<type>>;

    template <class type>
            static constexpr bool is_string_v = std::is_constructible_v<std::string, std::decay_t<type>>;

    template <class return_type, class func_type, class... args_type>
            static constexpr bool check_func_v = std::is_invocable_r_v<return_type, remove_all_qual<func_type>, args_type...>;

    template <class... types>
            static constexpr bool is_number_v = ((std::is_arithmetic_v<std::remove_pointer_t<types>> && !is_string_v<std::remove_pointer_t<types>>) && ...);

    template <std::size_t i, class tuple_type>
            using tuple_args_type_at = meta_types::remove_all_qual<std::tuple_element_t<i, meta_types::remove_all_qual<tuple_type>>>;

    template <std::size_t i, class tuple_type>
            using tuple_args_type_at_ = std::tuple_element_t<i, meta_types::remove_all_qual<tuple_type>>;

    template <class funcType, class... argsType>
    using return_type_t = std::invoke_result_t<meta_types::remove_all_qual<funcType>, argsType...>;

    template <class funcType, class... argsType>
            using create_functional_type = std::function<meta_types::return_type_t<funcType, argsType...>(argsType...)> ;

    template <class... types>
    struct are_same : std::true_type{};
    template <class T1, class... first_remaining_type, class T2, class... second_remaining_type>
    struct are_same<std::tuple<T1, first_remaining_type...>, std::tuple<T2, second_remaining_type...>> :
            std::conditional_t<std::is_same_v<T1, T2>, are_same<std::tuple<first_remaining_type...>, std::tuple<second_remaining_type...>>, std::false_type>{};

    template <class tuple_type>
    static bool is_tuple_default_initialized (tuple_type&& tup) {
        auto check = [] <class tuple_type_, std::size_t... i_> (tuple_type_&& tup_, std::index_sequence<i_...>) -> bool {
            auto check_at = [] <typename _type_> (_type_&& _arg_) -> bool {
                return _arg_ == _type_{};
            };
            return (check_at.operator()(
                    std::forward<tuple_args_type_at<i_, tuple_type_>>(std::get<i_>(std::forward<tuple_type_>(tup_))))
            && ...);
        };
        return check.operator()(std::forward<tuple_type>(tup), std::index_sequence_for<std::tuple_size<tuple_type>>{});
    }
}


#endif //TYPES_H
