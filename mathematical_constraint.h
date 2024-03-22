//
// Created by Harshavardhan Karnati on 16/03/2024.
//

#ifndef CONCEPTUAL_MATHEMATICAL_CONSTRAINT_H
#define CONCEPTUAL_MATHEMATICAL_CONSTRAINT_H

#include <iostream>
#include <vector>
#include <functional>
#include <tuple>

#include "unsupported/meta_types.h"

namespace aux {
    template <class returnType, class... argsType>
    class constraints_system {
    public:
        template <class constraintFuncTypes, class valueType>
        struct create_constraint {
            std::function<constraintFuncTypes> func;
            std::string operator_;
            valueType value;
            float tolerance = 0.00001F;

            template <class constraintFuncTypes_, class operatorType_, class valueType_, class toleranceType_>
            requires (meta_types::check_func_v<valueType_, constraintFuncTypes_, argsType...> && meta_types::is_string_v<operatorType_> && std::is_floating_point_v<toleranceType_>)
            create_constraint (constraintFuncTypes_&& IN_FUNC, operatorType_&& IN_OPERATOR, valueType_&& IN_VALUE, toleranceType_&& IN_TOLERANCE) {
                this->func = std::forward<constraintFuncTypes_>(IN_FUNC);
                this->operator_ = std::forward<operatorType_>(IN_OPERATOR);
                this->value = std::forward<valueType_>(IN_VALUE);
                this->tolerance = std::forward<toleranceType_>(IN_TOLERANCE);
            };
        };

        struct constraint_manager_base {
            bool constraints_on;
            returnType penalty{};
            virtual ~constraint_manager_base() = default;
            virtual void add_operators (const std::vector<std::string>& IN_OPERATORS) = 0;
            virtual void get_penalty (const std::tuple<argsType...> &IN_ARGS_TUPLE) = 0;
            virtual void add_tolerances (const std::vector<float>& IN_TOLERANCES) = 0;
        };

        template<class... constraintFuncTypes>
        struct constraint_manager final : constraint_manager_base {
            template <class constraintFuncType> using return_type_t = std::invoke_result_t<std::decay_t<constraintFuncType>, argsType...>;

            std::vector<std::function<void(argsType...)>> vector_of_constraints;
            std::tuple<std::shared_ptr<return_type_t<constraintFuncTypes>>...> return_tuple{};
            std::tuple<return_type_t<constraintFuncTypes>...> constraint_values;
            std::vector<std::string>operators;
            std::vector<float>tolerances;
            bool constraints_on = false;
            static constexpr std::size_t constraint_count = sizeof...(constraintFuncTypes);

            template<class... valueTypes>
            explicit constraint_manager (constraintFuncTypes... IN_FUNCS, valueTypes&&... IN_VALUES) {
                auto create = [this, &IN_FUNCS...] <std::size_t... i> (std::index_sequence<i...>) {
                    auto create_at = [this] <std::size_t i_, class constraintFuncType> (constraintFuncType &IN_FUNC) -> void {
                        using return_at = return_type_t<constraintFuncType>;
                        auto RETURN_POINTER = std::make_shared<return_at>();
                        std::get<i_>(this->return_tuple) = RETURN_POINTER;

                        auto constraint_at = [FUNC = IN_FUNC, RETURN_POINTER = RETURN_POINTER] (argsType... args) -> void {
                            *RETURN_POINTER = FUNC(args...);
                        };

                        this->vector_of_constraints.push_back(constraint_at);
                    };
                    (create_at.template operator()<i>(IN_FUNCS), ...);
                };
                create(std::index_sequence_for<constraintFuncTypes...>{});

                this->add_constraint_values(std::forward<valueTypes>(IN_VALUES)...);
            }

            template<class... valueTypes>
            void add_constraint_values (valueTypes&&... IN_VALUES) {
                std::tuple<meta_types::remove_all_qual<valueTypes>...> tuple = std::make_tuple(std::move(IN_VALUES)...);

                auto add = [this, &tuple] <std::size_t...i_> (std::index_sequence<i_...>) {
                    auto add_at = [this, &tuple] <std::size_t i> () {
                        try {
                            if(std::is_same_v<meta_types::tuple_args_type_at<i, decltype(this->constraint_values)>, meta_types::tuple_args_type_at<i, std::tuple<valueTypes...>>>) {
                                std::get<i>(this->constraint_values) = std::move(std::get<i>(tuple));
                            } else { throw std::runtime_error("Constraint values are not same as previously defined");}
                        } catch (std::exception& e) {
                            std::cerr << e.what() << std::endl;
                            std::cerr << "Skipping initialisation of value.. defaulted to" << std::get<i>(this->constraint_values) << std::endl;
                        }
                    };
                    (add_at.template operator()<i_>(),...);
                };
                add.operator()(std::index_sequence_for<valueTypes...>{});
            }

            void add_operators (const std::vector<std::string> &IN_OPERATORS) override {
                const std::size_t size = sizeof...(constraintFuncTypes);
                try {
                    if (size == IN_OPERATORS.size()) {
                        this->operators = IN_OPERATORS;
                    } else {throw std::runtime_error("Length of Operator vector does not match number of constraints");}
                } catch (std::exception &e) {
                    std::cerr << e.what() << std::endl;
                    std::cerr << "Declaring all operator to '<='" << std::endl;
                    this->operators.assign(size, "<=");
                }
            }

            void add_tolerances (const std::vector<float> &IN_TOLERANCES) override {
                const std::size_t size = sizeof...(constraintFuncTypes);
                try {
                    if (size == IN_TOLERANCES.size()) {
                        this->tolerances = IN_TOLERANCES;
                    } else {throw std::runtime_error("Length of Operator vector does not match number of constraints");}
                } catch (std::exception &e) {
                    std::cerr << e.what() << std::endl;
                    std::cerr << "Declaring all operator to '0.001f'" << std::endl;
                    this->tolerances.assign(size, 0.001F);
                }
            }

            auto get_constraint_violation (const auto& IN_OBTAINED, const auto& IN_REQUIRED, const std::string& IN_OPERATOR, const auto& IN_TOLERANCE) {
                std::remove_reference_t<std::decay_t<decltype(IN_REQUIRED)>> diff = IN_OBTAINED - IN_REQUIRED;
                if (IN_OPERATOR == "<" && diff >= 0 && std::abs(diff) > IN_TOLERANCE) {return std::abs(diff);}
                if (IN_OPERATOR == "<=" && diff > 0 && std::abs(diff) > IN_TOLERANCE) {return std::abs(diff);}
                if (IN_OPERATOR == ">" && diff <= 0 && std::abs(diff) > IN_TOLERANCE) {return std::abs(diff);}
                if (IN_OPERATOR == ">=" && diff < 0 && std::abs(diff) > IN_TOLERANCE) {return std::abs(diff);}
                if (IN_OPERATOR == "=" && std::abs(diff) > IN_TOLERANCE) {return std::abs(diff);}
                if (IN_OPERATOR == "!=" && std::abs(diff) < IN_TOLERANCE) {return std::numeric_limits<std::decay_t<decltype(IN_REQUIRED)>>::max();}
                return decltype(IN_REQUIRED){};
            }

            void get_penalty (const std::tuple<argsType...> &IN_ARGS_TUPLE) override {
                try {
                    this->penalty = {};

                    for (auto& each_constraint : this->vector_of_constraints) {std::apply(each_constraint, IN_ARGS_TUPLE);}

                    auto get_penalty_ = [this] <std::size_t... i> (std::index_sequence<i...>) {
                        auto get_penalty_at = [this] <std::size_t i_> () {
                            auto obt_value_at = *(std::get<i_>(this->return_tuple));
                            auto req_value_at = std::get<i_>(this->constraint_values);
                            this->penalty += static_cast<returnType>(this->get_constraint_violation(obt_value_at, req_value_at, this->operators[i_], this->tolerances[i_]));
                        };
                        (get_penalty_at.template operator()<i>(),...);
                    };
                    get_penalty_.operator()(std::index_sequence_for<constraintFuncTypes...>{});
                    const float slope = 1000000000.0F;
                    this->penalty = slope * this->penalty;
                } catch (std::exception &e) {
                    std::cerr << "Error while calculating constraint penalty..." << e.what() << std::endl;
                    std::cerr << "Ignoring constraints for current generation..." << std::endl;
                    this->penalty = {};
                }
            }
        };
    };
}


#endif //CONCEPTUAL_MATHEMATICAL_CONSTRAINT_H
