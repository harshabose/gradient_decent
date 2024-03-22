//
// Created by Harshavardhan Karnati on 14/03/2024.
//

#ifndef CONCEPTUAL_GRADIENT_DECENT_H
#define CONCEPTUAL_GRADIENT_DECENT_H

#define VERBOSITY 1

#if VERBOSITY
#define VERBOSE_PRINT(x)  do { \
    std::ostringstream oss; \
    oss << x; \
    std::cout << oss.str() << std::endl; \
} while (0)

#define VERBOSE_PRINT_(x)  do { \
    std::ostringstream oss; \
    oss << x; \
    std::cout << oss.str(); \
} while (0)

#define _VERBOSE_PRINT_(x) std::cout << x

#else
#define VERBOSE_PRINT(x)
#define VERBOSE_PRINT_(x)
#define _VERBOSE_PRINT_(x)
#endif

#include <iostream>
#include <sstream>
#include <functional>
#include <limits>
#include <tuple>
#include <utility>


#include "mathematical_constraint.h"
#include "unsupported/meta_types.h"

namespace gd {
    template <class returnType, class... argType>
    struct function_wrapper {
        template<class funcType>
        requires (meta_types::check_func_v<returnType, funcType, argType...>)
        explicit function_wrapper (funcType &&IN_FUNC) : function (std::forward<funcType>(IN_FUNC)) {}

        template<class tupleType>
        requires (std::is_same_v<meta_types::remove_all_qual<tupleType>, std::tuple<argType...>>)
        [[nodiscard]] returnType eval_func_at (tupleType &&IN_ARGS) const noexcept {
            return std::apply(this->function, std::forward<tupleType>(IN_ARGS));
        };

        template<class... argTypes_>
        [[nodiscard]] returnType eval_func_at (argTypes_&&... IN_ARGS) const noexcept {
            return this->eval_func_at(std::make_tuple(std::forward<argTypes_>(IN_ARGS)...));
        }

    private:
        std::function<returnType(argType...)> function;
    };





    template <class returnType, class... argType>
    class gradient_decent {
    public:

        gradient_decent () = default;

        template<class funcType_, class... argType_>
        explicit gradient_decent (funcType_ &&IN_FUNC, argType_ &&... IN_GUESS) {
            this->function = std::make_unique<gd::function_wrapper<returnType, argType...>>(
                    std::forward<funcType_>(IN_FUNC));
            this->optimal_point = std::make_tuple(std::forward<argType_>(IN_GUESS)...);
            this->optimal_val = this->eval_func_at(this->optimal_point);
            this->learning_rate = 1.0;
            this->finite_difference_step = 0.001;
            this->step_scales.fill(1.0);
            VERBOSE_PRINT("Gradient Decent instance created...");
        }

        virtual ~gradient_decent () = default;

        void set_max_eval (std::size_t IN_MAX_EVAL) noexcept {
            this->max_eval = IN_MAX_EVAL;
        };

        void set_tolerance (float IN_TOLERANCE) noexcept {
            this->tolerance = IN_TOLERANCE;
        };

        template<class tupleType>
        void add_lower_bounds (tupleType &&IN_LOWER_BOUNDS) {
            this->lower_bounds = std::forward<tupleType>(IN_LOWER_BOUNDS);
            if (this->check_point_bounds(indices_for_args{})) {
                std::cerr << "Initial guess is out-of-bounds. Use (public method) change_initial_guess()" << std::endl;
                throw std::runtime_error("Initial guess is out-of-bounds. Use (public method) change_initial_guess()");
            }
            VERBOSE_PRINT("Lower bounds set...");
        }

        template<class... argType_>
        void add_lower_bounds (argType_ &&... IN_LOWER_BOUNDS) {
            this->add_lower_bounds(std::make_tuple(std::forward<argType_>(IN_LOWER_BOUNDS)...));
        }

        template<class tupleType>
        void add_upper_bounds (tupleType &&IN_UPPER_BOUNDS) {
            this->upper_bounds = std::forward<tupleType>(IN_UPPER_BOUNDS);
            if (this->check_point_bounds(indices_for_args{})) {
                std::cerr << "Initial guess is out-of-bounds. Use (public method) change_initial_guess()" << std::endl;
                throw std::runtime_error("Initial guess is out-of-bounds. Use (public method) change_initial_guess()");
            }
            VERBOSE_PRINT("Upper bounds set...");
        }

        template<class... argType_>
        void add_upper_bounds (argType_ &&... IN_UPPER_BOUNDS) {
            this->add_upper_bounds(std::make_tuple(std::forward<argType_>(IN_UPPER_BOUNDS)...));
        }

        template<class type>
        void set_initial_learning_rate (type &&IN_RATE) noexcept {
            this->learning_rate = std::forward<type>(IN_RATE);
        }

        void toggle_classic_gradient_algo () {
            this->use_classic_gd = !this->use_classic_gd;
            if (this->use_classic_gd) { VERBOSE_PRINT("USING CLASSIC GRADIENT DECENT ALGORITHM..."); }
            else { VERBOSE_PRINT("USING SECANT SCALING APPROACH"); }
        }

        void toggle_learning_rate_scaling () {
            this->use_scaling = !this->use_scaling;
            if (this->use_scaling) {VERBOSE_PRINT("USING DERIVATIVE BASED LEARNING RATE SCALING");}
            else {VERBOSE_PRINT("NOT USING DERIVATIVE BASED LEARNING RATE SCALING");}
        }

        template <template <class, class, class...> class... createConstraintType, class constraintFuncType, typename valueType, class... argsType_>
        void add_constraints(createConstraintType<constraintFuncType, valueType, argsType_...>&... constraints) noexcept {
            this->constraints_on = true;
            this->constraint_manager_ = std::make_unique<aux::constraints_system<returnType, argType...>::template constraint_manager<decltype(constraints.func)...>>(std::move(constraints.func)..., (constraints.value)...);
            this->constraint_manager_->add_operators(std::vector<std::string>{constraints.operator_...});
            this->constraint_manager_->add_tolerances(std::vector<float>{constraints.tolerance...});
            VERBOSE_PRINT("Constraints ON");
            VERBOSE_PRINT("Added " << this->constraint_manager_.constraint_count << " constraints...");
        }

        std::pair<returnType, std::tuple<argType...>> perform_gradient_decent () {
            std::size_t eval = 0;
            do {
                this->old_optimal_point = this->optimal_point;
                VERBOSE_PRINT_("iteration @" << std::to_string(eval) << " with optimal val at " << this->optimal_val << " with point at ");
                this->verbose_print_tuple(this->optimal_point, indices_for_args{});
                this->step_scales.fill(1.0);
                this->calculate_derivatives_at(this->optimal_point);
                this->use_classic_gd ? this->back_tracking_algorithm(this->optimal_point) : this->step_forward(this->optimal_point);
                this->first_iteration_settings = false;
            } while (eval++ < this->max_eval && this->get_tolerance() > this->tolerance);


            if (eval >= this->max_eval && this->current_tolerance > this->tolerance) {
                throw std::runtime_error("Gradient descent failed to converge");
            }
            _VERBOSE_PRINT_("GD CONVERGED with optimal point at: ");
            this->verbose_print_tuple(this->optimal_point, indices_for_args{});
            VERBOSE_PRINT("with optimal value: " << this->optimal_val);
            VERBOSE_PRINT("Number of times fun called: " << this->func_call_count);

            return std::make_pair(this->optimal_val, this->optimal_point);
        }

    protected:
        using indices_for_args = std::index_sequence_for<argType...>;
        std::unique_ptr<gd::function_wrapper<returnType, argType...>> function;
        std::tuple<argType...> optimal_point;
        std::tuple<argType...> old_optimal_point;
        returnType optimal_val;
        std::size_t max_eval = 1000;
        returnType tolerance = 0.00001F;
        returnType current_tolerance = 0.002F;
        std::tuple<argType...> lower_bounds {};
        std::tuple<argType...> upper_bounds {};
        returnType learning_rate;
        returnType finite_difference_step;
        std::array<returnType, sizeof...(argType)> step_scales;
        std::tuple<argType...> derivatives {};
        std::tuple<argType...> derivative_high {};
        std::unique_ptr<typename aux::constraints_system<returnType, argType...>::constraint_manager_base> constraint_manager_;
        bool first_iteration_settings = true;
        bool constraints_on = false;
        bool use_classic_gd = false;
        bool use_scaling = false;
        std::size_t func_call_count = 0;

        template<class tupleType>
        requires(std::is_same_v<meta_types::remove_all_qual<tupleType>, std::tuple<argType...>>)
        returnType eval_func_at (tupleType&& IN_ARGS) noexcept {
            this->func_call_count++;
            if (this->constraints_on) {
                this->constraint_manager_->get_penalty(std::forward<tupleType>(IN_ARGS));
                return this->function->eval_func_at(std::forward<tupleType>(IN_ARGS)) + this->constraint_manager_->penalty;
            }
            return this->function->eval_func_at(std::forward<tupleType>(IN_ARGS));
        }

        void step_forward (std::tuple<argType...> IN_POINT) noexcept {
            this->optimal_point = std::move(this->bounds_projection(this->create_next_point(IN_POINT, indices_for_args{}), indices_for_args{}));
            returnType test_optimal = this->eval_func_at(this->optimal_point);
            if (test_optimal > this->optimal_val) {
                this->learning_rate += this->secant_learning_rate_scaling(test_optimal - this->optimal_val, this->optimal_val);
                this->learning_rate *= 0.5;
                this->optimal_point = std::move(this->bounds_projection(this->create_next_point(IN_POINT, indices_for_args{}), indices_for_args{}));
                this->optimal_val = this->eval_func_at(this->optimal_point);
            }
            else {
                this->current_tolerance = std::abs(this->optimal_val - test_optimal);
                this->optimal_val = test_optimal;
            }
        }

        void back_tracking_algorithm (std::tuple<argType...> IN_POINT) {
            std::size_t iterative_count = 0;
            std::size_t iterative_count_max = 1000;

            do {
                this->optimal_point = std::move(this->bounds_projection(this->create_next_point(IN_POINT, indices_for_args{}), indices_for_args{}));
                returnType test_optimal = this->eval_func_at(this->optimal_point);
                if (test_optimal > this->optimal_val) {
                    this->learning_rate *= 0.99;
                }
                else {
                    this->current_tolerance = std::abs(this->optimal_val - test_optimal);
                    this->optimal_val = test_optimal;
                    break;
                }
            } while (iterative_count++ < iterative_count_max);

            if (iterative_count >= iterative_count_max) {
                throw std::runtime_error("Cannot find next point using back-tracking algorithm");
            }
        }

        template <std::size_t i>
        void set_high_derivatives_helper () {
            if (std::abs(std::get<i>(this->derivatives)) > std::abs(std::get<i>(this->derivative_high))) {
                this->learning_rate = 1.0;
                std::get<i>(this->derivative_high) = std::get<i>(this->derivatives);
            }
        }

        template <std::size_t... i>
        void set_high_derivatives (std::index_sequence<i...>) {
            (this->set_high_derivatives_helper<i>(),...);
        }

        // TODO: Make this cleaner
        template<class tupleType, std::size_t... i>
        auto calculate_derivatives_at_helper (tupleType &&IN_TUPLE, std::index_sequence<i...>) noexcept {
            auto find_derivative_at = [this] <std::size_t i_, std::size_t... index> (tupleType &&IN_TUPLE_, std::index_sequence<index...>) -> meta_types::tuple_args_type_at<i_, tupleType> {
                meta_types::tuple_args_type_at<i_, tupleType> result{};
                try {
                    std::tuple<argType...> tuple_ = std::make_tuple((std::get<i>(IN_TUPLE_) * ((index == i_) ? (1.0F + this->finite_difference_step * this->step_scales.at(i_)) : 1.0F))...);
                    float factor = 1.0 / (std::get<i_>(IN_TUPLE_) * this->finite_difference_step * this->step_scales.at(i_));
                    result = (this->eval_func_at(tuple_) - this->optimal_val) * factor;
                } catch (std::exception &e) {
                    std::cerr << "Using backward finite element method instead" << std::endl;
                    std::tuple<argType...> tuple_ = std::make_tuple((std::get<i>(IN_TUPLE_) * ((index == i_) ? (1.0F - this->finite_difference_step * this->step_scales.at(i_)) : 1.0F))...);
                    float factor = 1.0 / (std::get<i_>(IN_TUPLE_) * this->finite_difference_step * this->step_scales.at(i_));
                    result = (this->eval_func_at(tuple_) - this->optimal_val) * factor;
                }
                return result;
            };
            return std::make_tuple(find_derivative_at.template operator()<i>(std::forward<tupleType>(IN_TUPLE), indices_for_args{})...);
        }

        template<class tupleType>
        std::tuple<argType...> calculate_derivatives_at (tupleType&& IN_POINT) {
            this->derivatives = this->calculate_derivatives_at_helper(std::forward<tupleType>(IN_POINT), indices_for_args{});
            this->set_high_derivatives(indices_for_args{});
            if (this->use_scaling) this->scale(this->step_scales.data(), indices_for_args{});
            return this->derivatives;
        }
        
        template <std::size_t... i>
        std::tuple<argType...>&& bounds_projection (std::tuple<argType...>&& IN_POINT, std::index_sequence<i...>) noexcept {
            (((std::get<i>(IN_POINT) < std::get<i>(this->lower_bounds)) ? std::get<i>(IN_POINT) = std::get<i>(this->lower_bounds) : std::get<i>(IN_POINT) = std::get<i>(IN_POINT)),...);
            (((std::get<i>(IN_POINT) > std::get<i>(this->upper_bounds)) ? std::get<i>(IN_POINT) = std::get<i>(this->upper_bounds) : std::get<i>(IN_POINT) = std::get<i>(IN_POINT)),...);
            return std::forward<std::tuple<argType...>&&>(IN_POINT);
        }
        
        template <std::size_t... i>
        bool check_point_bounds (std::index_sequence<i...>) {
            return ((std::get<i>(this->optimal_point) < std::get<i>(this->lower_bounds)) && ...) &&
                   ((std::get<i>(this->optimal_point) > std::get<i>(this->upper_bounds)) && ...);
        }

        template<class tupleType, std::size_t... i>
        void verbose_print_tuple (tupleType&& IN_TUPLE, std::index_sequence<i...>) {
#if VERBOSITY
            _VERBOSE_PRINT_("{");
            auto print_tuple_at = [] (tupleType&& IN_TUPLE_) -> void {
                ((std::cout << (i == 0? "" : ", ") << std::get<i>(IN_TUPLE_)),...);
            };
                print_tuple_at(std::forward<tupleType>(IN_TUPLE));
            VERBOSE_PRINT("}");
#endif
        }

        template <std::size_t... i>
        auto get_distance_tuple (std::tuple<argType...> &IN_FIRST_TUPLE, std::tuple<argType...> &IN_SEC_TUPLE, std::index_sequence<i...>) {
            auto sum = (((std::get<i>(IN_FIRST_TUPLE) - std::get<i>(IN_SEC_TUPLE)) * (std::get<i>(IN_FIRST_TUPLE) - std::get<i>(IN_SEC_TUPLE))) + ...);
            return std::sqrt(sum);
        }

        returnType get_tolerance () {
            returnType tol = this->current_tolerance + this->get_distance_tuple(this->optimal_point, this->old_optimal_point, indices_for_args{});
            return tol;
        }

        template <std::size_t... i>
        std::tuple<argType...> create_next_point (const std::tuple<argType...> &IN_POINT, std::index_sequence<i...>) noexcept {
            return std::move(std::make_tuple((std::get<i>(IN_POINT) - std::get<i>(this->derivatives) * this->learning_rate * this->step_scales.at(i))...));
        };

        returnType scale_function (const returnType IN_X, const returnType IN_FACTOR) {
            return std::sqrt(std::abs(IN_X / IN_FACTOR));
        }

        template <std::size_t... i>
        void scale (returnType* IN_TO_SCALE, std::index_sequence<i...>) {
            if (!(this->first_iteration_settings)) {
                ((*(IN_TO_SCALE + i) *= this->scale_function(std::get<i>(this->derivatives), std::get<i>(this->derivative_high))),...);
                ((*(IN_TO_SCALE + i) = *(IN_TO_SCALE + i) < this->tolerance ? this->tolerance : *(IN_TO_SCALE + i)),...);
            }
        }

        returnType secant_learning_rate_scaling (returnType IN_CURRENT_VAL, const returnType IN_REQUIRED_VAL) noexcept {

            returnType current_rate = 0.0;
            std::size_t iterative_count = 0;
            constexpr std::size_t iterative_max = 100;
            returnType new_rate = current_rate - 0.5;
            returnType new_val;
            auto find_new_rate = [&new_rate, &current_rate, &new_val, &IN_CURRENT_VAL] () -> returnType {
                return new_rate - new_val * (new_rate - current_rate) / (new_val - IN_CURRENT_VAL);
            };
            auto find_new_val = [&IN_REQUIRED_VAL, this] <std::size_t... i> (returnType &IN_RATE, std::index_sequence<i...>) {
                return this->eval_func_at(std::make_tuple((std::get<i>(this->optimal_point) - IN_RATE * std::get<i>(this->step_scales) * std::get<i>(this->derivatives))...)) - IN_REQUIRED_VAL;
            };

            do {
                new_val = find_new_val(new_rate, indices_for_args{});
                current_rate = std::exchange(new_rate, find_new_rate());
                IN_CURRENT_VAL = find_new_val(current_rate, indices_for_args {});
            } while (std::abs(new_rate - current_rate) > 0.001 && iterative_count++ < iterative_max);
            return new_rate;
        }
    };
}

#endif //CONCEPTUAL_GRADIENT_DECENT_H
