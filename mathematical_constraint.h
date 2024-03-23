/**
 * @file mathematical_constraint.h
 * @brief Header file containing the declaration of the conceptual mathematical constraint system.
 *
 * This file declares a mathematical constraint system that can be used in optimisation algorithms.
 * It provides functionalities to define constraints, evaluate constraint violations, and penalise
 * violations in an optimisation process.
 *
 * @author Harshavardhan Karnati
 * @date 16/03/2024
 */

#ifndef CONCEPTUAL_MATHEMATICAL_CONSTRAINT_H
#define CONCEPTUAL_MATHEMATICAL_CONSTRAINT_H

#include <iostream>
#include <vector>
#include <functional>
#include <tuple>

#include "unsupported/meta_types.h"

namespace aux {
    /**
     * @brief Template class for managing a system of mathematical constraints.
     *
     * This class provides functionalities for creating and managing mathematical constraints.
     * It allows users to define constraint functions, specify constraint values, operators,
     * and tolerances, and evaluate constraint violations.
     *
     * @tparam returnType The return type of the constraint function.
     * @tparam argsType The argument types of the constraint function.
     */
    template <class returnType, class... argsType>
    class constraints_system {
    public:
        /**
         * @brief Template struct for creating a mathematical constraint.
         *
         * This struct provides a mechanism to create a mathematical constraint with specified properties,
         * including the constraint function, operator, value, and tolerance.
         *
         * @tparam constraintFuncTypes The type of the constraint function.
         * @tparam valueType The type of the constraint value.
         */
        template <class constraintFuncTypes, class valueType>
        struct create_constraint {
            /**
             * @brief The constraint function.
             */
            std::function<constraintFuncTypes> func;
            /**
             * @brief The operator used for the constraint.
             */
            std::string operator_;
            /**
             * @brief The value against which the constraint is evaluated.
             */
            valueType value;
            /**
             * @brief The tolerance for constraint violation.
             */
            float tolerance = 0.00001F;

            /**
             * @brief Constructor for creating a constraint.
             *
             * Constructs a constraint with the provided function, operator, value, and tolerance.
             *
             * @tparam constraintFuncTypes_ The decayed type of the constraint function.
             * @tparam operatorType_ The decayed type of the operator.
             * @tparam valueType_ The decayed type of the value.
             * @tparam toleranceType_ The decayed type of the tolerance.
             *
             * @param IN_FUNC The constraint function.
             * @param IN_OPERATOR The operator used for the constraint.
             * @param IN_VALUE The value against which the constraint is evaluated.
             * @param IN_TOLERANCE The tolerance for constraint violation.
             *
             * @code{.cpp}
             * // example
             * auto constraint1_ = aux::constraints_system<returnType, argsType...>::create_constraint(constraint_function_1, "&#x3C;", 9, 0.001f);
             * @endcode
             */
            template <class constraintFuncTypes_, class operatorType_, class valueType_, class toleranceType_>
            requires (meta_types::check_func_v<valueType_, constraintFuncTypes_, argsType...> && meta_types::is_string_v<operatorType_> && std::is_floating_point_v<toleranceType_>)
            create_constraint (constraintFuncTypes_&& IN_FUNC, operatorType_&& IN_OPERATOR, valueType_&& IN_VALUE, toleranceType_&& IN_TOLERANCE) {
                this->func = std::forward<constraintFuncTypes_>(IN_FUNC);
                this->operator_ = std::forward<operatorType_>(IN_OPERATOR);
                this->value = std::forward<valueType_>(IN_VALUE);
                this->tolerance = std::forward<toleranceType_>(IN_TOLERANCE);
            };
        };

        /**
         * @brief Abstract base class for managing mathematical constraints.
         *
         * The constraint_manager_base struct defines an abstract base class for managing mathematical constraints.
         * It provides interfaces for adding operators, calculating penalty, and adding tolerances for constraints.
         * The primary purpose of this abstract is to allow polymorphism for ease-of-use while implementing a
         * constraint for an optimisation problem.
         */
        struct constraint_manager_base {
            /**
             * @brief Flag indicating if constraints are enabled.
             */
            bool constraints_on = false;
            /**
             * @brief Penalty associated with constraint violations.
             */
            returnType penalty{};


            /**
             * @brief Virtual destructor for constraint_manager_base.
             *
             * The virtual destructor ensures proper destruction of derived classes.
             */
            virtual ~constraint_manager_base() = default;

            /**
             * @brief Virtual function to add operators for constraints.
             *
             * This function is used to add operators for the constraints.
             *
             * @param IN_OPERATORS The vector of operator strings to be added.
             */
            virtual void add_operators (const std::vector<std::string>& IN_OPERATORS) = 0;

            /**
             * @brief Virtual function to calculate penalty for constraint violations.
             *
             * This function calculates the penalty associated with constraint violations based on the input arguments.
             *
             * @param IN_ARGS_TUPLE The tuple containing input arguments for evaluating constraints.
             */
            virtual void get_penalty (const std::tuple<argsType...> &IN_ARGS_TUPLE) = 0;

            /**
             * @brief Virtual function to add tolerances for constraints.
             *
             * This function is used to add tolerances for the constraints.
             *
             * @param IN_TOLERANCES The vector of tolerance values to be added.
             */
            virtual void add_tolerances (const std::vector<float>& IN_TOLERANCES) = 0;
        };


        /**
         * @brief Manager for handling mathematical constraints.
         *
         * The constraint_manager struct provides functionality for managing mathematical constraints
         * in an optimization problem. It serves as a container for storing constraint functions,
         * their corresponding values, operators, and tolerances.
         *
         * @tparam constraintFuncTypes Variadic template parameter representing constraint function types.
         * @details The `constraint_manager` struct is a final class derived from `constraint_manager_base`.
         * It allows users to add constraint functions and their values, set operators, and tolerances for each constraint.
         *
         * Example Usage:
         * @code{.cpp}
         * // Define constraint functions
         * auto constraint_function_1 = [] (double x, double y, some_pointer* ptr) -> double {return x*x + y*y;}
         * auto constraint_function_2 = [] (double x, double y, some_pointer* ptr) -> double {return x + y;}
         *
         * // Define constraint manager
         * auto manager = std::make_unique.template <aux::constraint_manager_base>();
         *
         * // Create Constraint
         * auto constraint1_ = aux::constraints_system<returnType, argsType...>::create_constraint(constraint_function_1, "&#x3C;", 9, 0.001);
         * auto constraint2_ = aux::constraints_system<returnType, argsType...>::create_constraint(constraint_function_2, "&#x3C;", 4, 0.001);
         *
         * // Send all constraints
         * manager = std::make_unique.template <aux::constraints_system<returnType, argType...>::constraint_manager<decltype(constraint1.func), decltype(constraint2_.func)>>>(constraint1_.fun, constraint2_.fun, constraint1_.value, constraint2_.value)
         * @endcode
         *
         * The struct includes member functions for adding constraint values, operators, calculating penalties,
         * and checking constraint violations.
         */
        template<class... constraintFuncTypes>
        struct constraint_manager final : constraint_manager_base {
            /**
             * Alias for the return type of a constraint function.
             */
            template <class constraintFuncType> using return_type_t = std::invoke_result_t<std::decay_t<constraintFuncType>, argsType...>;

            std::vector<std::function<void(argsType...)>> vector_of_constraints;
            std::tuple<std::shared_ptr<return_type_t<constraintFuncTypes>>...> return_tuple{};
            std::tuple<return_type_t<constraintFuncTypes>...> constraint_values;
            std::vector<std::string>operators;
            std::vector<float>tolerances;
            bool constraints_on = false;
            static constexpr std::size_t constraint_count = sizeof...(constraintFuncTypes);

            /**
             * @brief Constructor for the constraint manager.
             *
             * Constructs the constraint manager with provided constraint functions and values.
             *
             * @param IN_FUNCS Constraint functions.
             * @param IN_VALUES Constraint values.
             */
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

            /**
             * @brief Adds constraint values to the manager.
             *
             * Adds constraint values to the manager based on provided input values.
             *
             * @param IN_VALUES Input values for constraints.
             */
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

            /**
             * @brief Adds operators to the manager.
             *
             * Adds constraint operators to the manager.
             *
             * @param IN_OPERATORS Vector of operators.
             */
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

            /**
             * @brief Adds tolerances to the manager.
             *
             * Adds constraint tolerances to the manager.
             *
             * @param IN_TOLERANCES Vector of tolerances.
             */
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

            /**
             * @brief Calculates constraint violation.
             *
             * Calculates the violation of a constraint based on obtained and required values,
             * operator, and tolerance.
             *
             * @param IN_OBTAINED Obtained value.
             * @param IN_REQUIRED Required value.
             * @param IN_OPERATOR Constraint operator.
             * @param IN_TOLERANCE Constraint tolerance.
             * @return The violation value.
             * */
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

            /**
             * @brief Calculates penalty for constraint violation.
             *
             * Calculates the penalty for constraint violation based on obtained and required values,
             * operator, and tolerance.
             *
             * @param IN_ARGS_TUPLE Tuple of input arguments.
             */
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
