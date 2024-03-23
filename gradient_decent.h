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

/**
 * @brief Namespace for gradient descent optimisation utilities.
 *
 * The gd namespace encapsulates utilities and structures related to gradient descent optimisation.
 * It provides functionality for wrapping objective functions and performing gradient descent optimisation.
 */
namespace gd {
    /**
     * @brief Wrapper for an objective function.
     *
     * The function_wrapper struct provides a wrapper for an objective function
     * with specified return type and argument types with special checks and safety guards.
     * It allows evaluation of the objective function at different argument values.
     *
     * @tparam returnType The return type of the objective function.
     * @tparam argType The argument types of the objective function.
     * */
    template <class returnType, class... argType>
    struct function_wrapper {
        /**
         * @brief Constructor for the function wrapper.
         *
         * Constructs a function wrapper instance with the provided objective function.
         *
         * @tparam funcType The type of the objective function.
         * @param IN_FUNC The objective function to be wrapped.
         *
         * @details
         * The constructor initialises the function wrapper with the provided objective function.
         * It ensures that the provided function matches the expected return type and argument types.
         * If the function does not match, a compilation error occurs.
         */
        template<class funcType>
        requires (meta_types::check_func_v<returnType, funcType, argType...>)
        explicit function_wrapper (funcType &&IN_FUNC) : function (std::forward<funcType>(IN_FUNC)) {}

        /**
         * @brief Evaluates the objective function at the specified arguments.
         *
         * Evaluates the objective function at the specified arguments and returns the result.
         *
         * @tparam tupleType The type of the tuple containing arguments.
         * @param IN_ARGS The tuple containing arguments at which the function is evaluated.
         * @return The result of evaluating the objective function at the specified arguments.
         *
         * @details
         * The eval_func_at method evaluates the objective function at the specified arguments
         * provided as a tuple. It applies the objective function to the tuple using std::apply
         * and returns the result.
         *
         * @note Marked as noexcept to allow optimisation (if allowed by function_wrapper::function).
         */
        template<class tupleType>
        requires (std::is_same_v<meta_types::remove_all_qual<tupleType>, std::tuple<argType...>>)
        [[nodiscard]] returnType eval_func_at (tupleType &&IN_ARGS) const noexcept {
            return std::apply(this->function, std::forward<tupleType>(IN_ARGS));
        };

        /**
         * @brief Evaluates the objective function at the specified arguments.
         *
         * This is an overload to allow a convenient way to input as variadic arguments.
         * Evaluates the objective function at the specified arguments and returns the result.
         *
         * @tparam argTypes_ The types of the arguments.
         * @param IN_ARGS The arguments at which the function is evaluated.
         * @return The result of evaluating the objective function at the specified arguments.
         *
         * @details
         * The eval_func_at method evaluates the objective function at the specified arguments
         * provided individually. It creates a tuple from the individual arguments and delegates
         * the evaluation to the eval_func_at method that accepts a tuple.
         *
         * @note Marked as noexcept to allow optimisation (if allowed by function_wrapper::function).
         */
        template<class... argTypes_>
        [[nodiscard]] returnType eval_func_at (argTypes_&&... IN_ARGS) const noexcept {
            return this->eval_func_at(std::make_tuple(std::forward<argTypes_>(IN_ARGS)...));
        }
    private:
        std::function<returnType(argType...)> function; ///< The objective function to be wrapped.
    };




    /**
     * @brief Class for gradient descent optimisation algorithm.
     *
     * The gradient_descent class implements the gradient descent optimisation algorithm,
     * which iteratively minimises an objective function by adjusting optimisation variables.
     * It provides methods to set optimisation parameters, add constraints, and perform optimisation.
     *
     * @tparam returnType The type of the objective function's return value.
     * @tparam argType The types of the optimisation variables.
     *
     * @details
     * The gradient_descent class enables optimisation of objective functions with respect
     * to multiple optimisation variables using the gradient descent algorithm.
     * It provides flexibility to specify custom optimisation parameters, such as learning rate,
     * maximum number of iterations, and tolerance for convergence.
     *
     * The class supports adding constraints on optimization variables and toggling between
     * different optimisation algorithms, such as classic gradient descent and secant-based scaling.
     * Additionally, momentum rotation of derivatives and derivative based scaling can be employed
     * as needed.
     *
     * @note Ensure that the objective function and optimisation variables are compatible
     * with the specified types. Experiment with different optimisation parameters
     * and algorithms to achieve optimal convergence and performance.
     */
    template <class returnType, class... argType>
    class gradient_decent {
    public:

        /**
         * @brief Default constructor for gradient descent.
         * Constructs a gradient descent optimizer with NSDMI default parameters.
         */
        gradient_decent () = default;

        /**
         * @brief Constructor for the gradient descent optimiser.
         *
         * Constructs a gradient descent optimiser instance with the specified objective function and initial guess.
         * The optimiser iteratively minimises the objective function by adjusting the optimisation variables.
         *
         * @tparam funcType_ The type of the objective function.
         * @tparam argType_ The types of the optimisation variables.
         * @param IN_FUNC The objective function to be minimised.
         * @param IN_GUESS The initial guess for the optimisation variables.
         *
         * @details
         * The constructor initialises the gradient descent optimiser instance with the provided objective function
         * and initial guess for the optimisation variables.
         *
         * It ensures that the types of the optimisation variables match the types expected by the objective function
         * and that the objective function returns a value compatible with the specified return type.
         * If the types do not match, a compilation error occurs.
         *
         * The constructor creates a function wrapper for the objective function and sets it as a member variable.
         * It also initialises the optimal point with the provided initial guess, evaluates the objective function
         * at the initial guess point to obtain the initial value, and sets default values for learning rate (1.0),
         * finite difference step (0.001), and step scales (1.0).
         *
         * @note Ensure that the objective function and initial guess are compatible with the specified types.
         * Experiment with different initial guesses and optimisation parameters to achieve optimal convergence.
         */
        template<class funcType_, class... argType_>
        requires(meta_types::are_same<argType_..., argType...>::value && meta_types::check_func_v<returnType, funcType_, argType_...>)
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

        /**
         * @brief Virtual destructor for gradient descent.
         * The virtual destructor for the gradient descent class is declared as default.
         * It ensures proper cleanup of resources when derived classes are destroyed.
         */
        virtual ~gradient_decent () = default;

        /**
         * @brief Sets the maximum number of evaluations for optimisation.
         *
         * This method sets the maximum number of evaluations allowed during the optimisation process.
         * The maximum number of evaluations determines the termination criteria for the optimisation algorithm.
         *
         * @param IN_MAX_EVAL The maximum number of evaluations for optimisation.
         *
         * @details
         * The method sets the `max_eval` member variable to the provided maximum number of evaluations.
         * Once the number of evaluations reaches this limit, the optimisation algorithm terminates,
         * regardless of whether the convergence criteria are met.
         *
         * @note Adjust the maximum number of evaluations based on the desired optimisation runtime and resources.
         * Setting a larger value allows for more iterations, potentially leading to better convergence,
         * but may also increase the computational cost.
         * Conversely, setting a smaller value limits the number of iterations, reducing the computational cost,
         * but may also affect convergence quality.
         */
        void set_max_eval (std::size_t IN_MAX_EVAL) noexcept {
            this->max_eval = IN_MAX_EVAL;
        };

        /**
         * @brief Sets the tolerance for convergence criteria.
         *
         * This method sets the tolerance value used as a convergence criteria in the optimisation process.
         * The tolerance value determines the acceptable difference between consecutive iterations' objective function values.
         *
         * @param IN_TOLERANCE The tolerance value for convergence criteria.
         *
         * @details
         * The method sets the `tolerance` member variable to the provided tolerance value.
         * The tolerance value represents the acceptable difference between consecutive iterations' objective function values,
         * indicating convergence if the difference falls below this threshold.
         *
         * @note Adjust the tolerance value based on the desired convergence criteria.
         * A smaller tolerance value indicates stricter convergence criteria, leading to more accurate results but potentially
         * requiring more iterations for convergence.
         * Conversely, a larger tolerance value allows for looser convergence criteria, resulting in faster convergence but
         * potentially sacrificing accuracy.
         */
        void set_tolerance (returnType IN_TOLERANCE) noexcept {
            this->tolerance = IN_TOLERANCE;
        };

        /**
         * @brief Adds lower bounds for the optimisation variables.
         *
         * This method adds lower bounds for the optimisation variables. The lower bounds
         * constrain the search space during the optimisation process.
         *
         * @tparam tupleType The type of the tuple containing lower bounds for each optimisation variable.
         * @param IN_LOWER_BOUNDS The tuple containing lower bounds for each optimisation variable.
         *
         * @details
         * The method sets the `lower_bounds` member variable to the provided tuple containing lower bounds.
         * It verifies whether the initial guess point lies within the specified lower bounds. If the initial guess
         * is out of bounds, an error message is printed, and a runtime error is thrown.
         *
         * The method ensures that the type of the tuple containing lower bounds matches the type of the tuple
         * representing the optimisation variables using a requires clause. If the types do not match, a compilation
         * error occurs.
         *
         * @note The lower bounds restrict the search space of the optimisation variables.
         * Ensure that the initial guess point lies within the specified lower bounds to avoid runtime errors.
         * Use the `change_initial_guess()` method to update the initial guess point if needed.
         */
        template<class tupleType>
        requires(meta_types::are_tuples_same_v<tupleType, std::tuple<argType...>>)
        void add_lower_bounds (tupleType &&IN_LOWER_BOUNDS) {
            this->lower_bounds = std::forward<tupleType>(IN_LOWER_BOUNDS);
            if (this->check_point_bounds(indices_for_args{})) {
                std::cerr << "Initial guess is out-of-bounds. Use (public method) change_initial_guess()" << std::endl;
                throw std::runtime_error("Initial guess is out-of-bounds. Use (public method) change_initial_guess()");
            }
            VERBOSE_PRINT("Lower bounds set...");
        }

        /**
         * @brief Adds lower bounds for the optimisation variables (overload).
         *
         * This method is an overload of the add_lower_bounds method and provides a convenient way
         * to specify lower bounds for the optimisation variables as individual arguments.
         *
         * @tparam argType_ The types of the lower bounds for each optimisation variable.
         * @param IN_LOWER_BOUNDS The lower bounds for each optimisation variable as individual arguments.
         *
         * @details
         * The method forwards the individual lower bound arguments to the original add_lower_bounds method
         * by creating a tuple from the arguments using std::make_tuple and forwarding it to the original method.
         * It ensures that the types of the lower bounds match the types of the optimisation variables.
         * If the types do not match, a compilation error occurs.
         *
         * @note This overload provides a convenient way to specify lower bounds for the optimisation variables
         * without explicitly creating a tuple.
         * Ensure that the types and number of lower bounds match the types and number of optimisation variables.
         */
        template<class... argType_>
        requires(meta_types::are_same<argType_..., argType...>::value)
        void add_lower_bounds (argType_ &&... IN_LOWER_BOUNDS) {
            this->add_lower_bounds(std::make_tuple(std::forward<argType_>(IN_LOWER_BOUNDS)...));
        }

        /**
         * @brief Adds upper bounds for the optimisation variables.
         *
         * This method adds upper bounds for the optimisation variables. The upper bounds
         * constrain the search space during the optimisation process.
         *
         * @tparam tupleType The type of the tuple containing upper bounds for each optimisation variable.
         * @param IN_UPPER_BOUNDS The tuple containing upper bounds for each optimisation variable.
         *
         * @details
         * The method sets the `upper_bounds` member variable to the provided tuple containing upper bounds.
         * It verifies whether the initial guess point lies within the specified upper bounds. If the initial guess
         * is out of bounds, an error message is printed, and a runtime error is thrown.
         *
         * The method ensures that the type of the tuple containing upper bounds matches the type of the tuple
         * representing the optimisation variables using a requires clause. If the types do not match, a compilation
         * error occurs.
         *
         * @note The upper bounds restrict the search space of the optimisation variables.
         * Ensure that the initial guess point lies within the specified upper bounds to avoid runtime errors.
         * Use the `change_initial_guess()` method to update the initial guess point if needed.
         */
        template<class tupleType>
        requires(meta_types::are_tuples_same_v<tupleType, std::tuple<argType...>>)
        void add_upper_bounds (tupleType &&IN_UPPER_BOUNDS) {
            this->upper_bounds = std::forward<tupleType>(IN_UPPER_BOUNDS);
            if (this->check_point_bounds(indices_for_args{})) {
                std::cerr << "Initial guess is out-of-bounds. Use (public method) change_initial_guess()" << std::endl;
                throw std::runtime_error("Initial guess is out-of-bounds. Use (public method) change_initial_guess()");
            }
            VERBOSE_PRINT("Upper bounds set...");
        }

        /**
         * @brief Adds upper bounds for the optimisation variables (overload).
         *
         * This method is an overload of the add_upper_bounds method and provides a convenient way
         * to specify upper bounds for the optimisation variables as individual arguments.
         *
         * @tparam argType_ The types of the upper bounds for each optimisation variable.
         * @param IN_UPPER_BOUNDS The upper bounds for each optimisation variable as individual arguments.
         *
         * @details
         * The method forwards the individual upper bound arguments to the original add_upper_bounds method
         * by creating a tuple from the arguments using std::make_tuple and forwarding it to the original method.
         * It ensures that the types of the upper bounds match the types of the optimization variables.
         * If the types do not match, a compilation error occurs.
         *
         * @note This overload provides a convenient way to specify upper bounds for the optimisation variables
         * without explicitly creating a tuple.
         * Ensure that the types and number of upper bounds match the types and number of optimisation variables.
         */
        template<class... argType_>
        requires(meta_types::are_same<argType_..., argType...>::value)
        void add_upper_bounds (argType_ &&... IN_UPPER_BOUNDS) {
            this->add_upper_bounds(std::make_tuple(std::forward<argType_>(IN_UPPER_BOUNDS)...));
        }

        /**
         * @brief Sets the initial learning rate for optimisation.
 *
         * This method sets the initial learning rate to be used in the optimisation process.
         * The learning rate determines the step size during gradient descent iterations.
 *
         * @tparam type The type of the initial learning rate.
         * @param IN_RATE The initial learning rate value.
 *
         * @details
         * The method sets the `learning_rate` member variable to the provided initial learning rate value.
         * The type of the initial learning rate is deduced based on the argument passed to the function.
         * The method ensures that the type of the initial learning rate matches the return type of the objective
         * function using a requires clause. If the types do not match, a compilation error occurs.
 *
         * @note The initial learning rate affects the convergence speed and stability of the optimisation process.
         * Experiment with different initial learning rates to achieve optimal performance.
 */
        template<class type>
        requires (std::is_same_v<meta_types::remove_all_qual<type>, returnType>)
        void set_initial_learning_rate (type &&IN_RATE) noexcept {
            this->learning_rate = std::forward<type>(IN_RATE);
        }

        /**
         * @brief Toggles between classic gradient descent algorithm and new approach.
         *
         * This method toggles between using the classic gradient descent algorithm and a new approach
         * called "Gradient Descent Algorithm with Secant Method Scaling". The choice of algorithm affects
         * the optimisation process and determines how the gradient descent is performed.
         *
         * @details
         * The method toggles the `use_classic_gd` flag to switch between using the classic gradient descent
         * algorithm and the new approach. If the classic gradient descent algorithm is selected, a message
         * indicating its usage is printed. If the new approach is selected, a message indicating its usage
         * is printed. The verbosity flag determines whether these messages are printed.
         *
         * @note This method impacts the optimization process and should be used based on experimentation
         * and analysis of the problem characteristics. The choice of algorithm may affect convergence
         * speed and solution quality.
         */
        void toggle_classic_gradient_algo () {
            this->use_classic_gd = !this->use_classic_gd;
            if (this->use_classic_gd) { VERBOSE_PRINT("USING CLASSIC GRADIENT DECENT ALGORITHM..."); }
            else { VERBOSE_PRINT("USING SECANT SCALING APPROACH"); }
        }

        /**
         * @brief Toggles derivative-based scaling for learning rate and finite difference.
         *
         * This method toggles the use of derivative-based scaling for adjusting the learning rate
         * and finite difference in the optimisation process. When derivative scaling is enabled,
         * the learning rate and finite difference are adjusted based on the derivatives of the objective
         * function. When disabled, fixed values are used for the learning rate and finite difference. (They
         * might be controlled by other parameters)
         *
         * @details
         * The method toggles the `use_scaling` flag to enable or disable derivative-based scaling.
         * If derivative scaling is enabled, a message indicating its usage is printed. If disabled,
         * a message indicating its non-usage is printed. The verbosity flag determines whether these
         * messages are printed.
         *
         * @note By default this is off. This method affects the optimisation process and should be used with caution.
         * It changes the behavior of the learning rate and finite difference calculation.
         * Use this method to experiment with different scaling approaches based on the problem characteristics.
         */
        void toggle_derivative_scaling () {
            this->use_scaling = !this->use_scaling;
            if (this->use_scaling) {VERBOSE_PRINT("USING DERIVATIVE BASED LEARNING RATE SCALING");}
            else {VERBOSE_PRINT("NOT USING DERIVATIVE BASED LEARNING RATE SCALING");}
        }

        /**
         * @brief Adds constraints to the optimisation problem.
         *
         * This method adds mathematical constraints to the optimisation problem. Constraints can be of various types
         * and are specified using template parameters. The constraints are stored in the constraint manager
         * (an object of aux::constraints_system::constraint_manager) for handling during the optimisation process.
         * To create a constraint use aux::constraints_system::create_constraint
         *
         * @tparam createConstraintType... Variadic template parameter for the types of constraint creation classes.
         * @tparam constraintFuncType     The type of the constraint function defined in createConstraintType.
         * @tparam valueType              The type of the constraint value defined in createConstraintType.
         * @tparam argsType_...           Variadic template parameter for the types of arguments of the
         *                                constraint function defined in createConstraintType.
         * @param constraints...          Variadic parameter pack of constraint objects defined in createConstraintType.
         *
         * @details
         * The method sets the `constraints_on` flag to true to indicate that constraints are active.
         * It creates a unique pointer to a constraint manager, passing the constraint function, value,
         * and arguments from each constraint object. The method then adds operators and tolerances to the
         * constraint manager based on the provided constraints. If verbosity is enabled, it prints a message
         * indicating that constraints are activated and the number of constraints added.
         *
         * @param constraints... Variadic parameter pack of constraint objects containing:
         * - `func`: The constraint function.
         * - `value`: The value associated with the constraint.
         * - `operator_`: The operator used in the constraint.
         * - `tolerance`: The tolerance value for the constraint.
         *
         * @code{.cpp}
         * //example
         * auto constraint1_ = aux::constraints_system<returnType, argsType...>::create_constraint(constraint_function_1, "&#x3C;", 9, 0.001f);
         * @endcode
         *
         * @note Constraints must be added before performing the optimisation.
         * The method assumes that the provided constraint objects have member variables:
         * `func`, `value`, `operator_`, and `tolerance`.
         * */
        template <template <class, class, class...> class... createConstraintType, class constraintFuncType, typename valueType, class... argsType_>
        void add_constraints(createConstraintType<constraintFuncType, valueType, argsType_...>&... constraints) noexcept {
            this->constraints_on = true;
            this->constraint_manager_ = std::make_unique<aux::constraints_system<returnType, argType...>::template constraint_manager<decltype(constraints.func)...>>(std::move(constraints.func)..., (constraints.value)...);
            this->constraint_manager_->add_operators(std::vector<std::string>{constraints.operator_...});
            this->constraint_manager_->add_tolerances(std::vector<float>{constraints.tolerance...});
            VERBOSE_PRINT("Constraints ON");
            VERBOSE_PRINT("Added " << this->constraint_manager_.constraint_count << " constraints...");
        }

        /**
         * @brief Performs gradient descent optimization.
         *
         * This method performs gradient descent optimisation to find the optimal point
         * for the given function. It iteratively updates the current point, function value
         * until convergence and derivatives until convergence or until the maximum number
         * of iterations is reached.
         *
         * @tparam returnType The return type of the objective function.
         * @tparam argType    The argument types of the objective function.
         * @return A pair containing the optimal value and the optimal point.
         *
         * @details The method initializes an evaluation counter and enters a do-while loop to perform
         * gradient descent iterations. Within each iteration:
         * <ul>
         * <li> The old optimal point is updated.
         * <li> The iteration details are printed if verbosity is enabled.
         * <li> Step scales are reset to 1.0.
         * <li> Derivatives are calculated at the optimal point.
         * <li> Either classic gradient descent with backtracking or step forward algorithm with secant method scaling
         * is applied.
         * </ul>
         * The loop continues until the evaluation counter exceeds the maximum evaluation count
         * or the tolerance condition is met.
         *
         * If gradient descent fails to converge within the specified maximum evaluation count
         * and tolerance, a runtime error is thrown.
         *
         * After convergence, the optimal point and value are printed if verbosity is enabled.
         *
         * @note The algorithm uses classic gradient descent with backtracking or step forward algorithm with secant
         * method scaling based on the `use_classic_gd` flag.
         */
        std::pair<returnType, std::tuple<argType...>> perform_gradient_decent () {
            std::size_t eval = 0;
            do {
                this->old_optimal_point = this->optimal_point;
                VERBOSE_PRINT_("iteration @" << std::to_string(eval) << " with optimal val at " << this->optimal_val << " with point at ");
                this->verbose_print_tuple(this->optimal_point, indices_for_args{});
                this->step_scales.fill(1.0);
                this->calculate_derivatives_at(this->optimal_point);
                this->use_classic_gd ? this->step_forward_with_back_tracking(this->optimal_point) : this->step_forward_with_secant_method(this->optimal_point);
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
        /**
         * @brief Indices for the argument types.
         */
        using indices_for_args = std::index_sequence_for<argType...>;
        /**
         * @brief Unique pointer to the function wrapper.
         */
        std::unique_ptr<gd::function_wrapper<returnType, argType...>> function;
        /**
         * @brief Tuple representing the optimal point.
         */
        std::tuple<argType...> optimal_point;
        /**
         * @brief Tuple representing the old optimal point.
         */
        std::tuple<argType...> old_optimal_point;
        /**
         * @brief Value of the objective function at the optimal point.
         */
        returnType optimal_val;
        /**
         * @brief Maximum number of evaluations for optimization (default: 1000).
         */
        std::size_t max_eval = 1000;
        /**
         * @brief Tolerance for convergence criteria (default: 0.00001F).
         */
        returnType tolerance = 0.00001F;
        /**
         * @brief Current tolerance value (default: 0.002F).
         */
        returnType current_tolerance = 0.002F;
        /**
         * @brief Tuple representing the lower bounds for optimization variables.
         */
        std::tuple<argType...> lower_bounds {};
        /**
         * @brief Tuple representing the upper bounds for optimization variables.
         */
        std::tuple<argType...> upper_bounds {};
        /**
         * @brief Learning rate for gradient descent optimization.
         */
        returnType learning_rate;
        /**
         * @brief Finite difference step for numerical differentiation.
         */
        returnType finite_difference_step;
        /**
         * @brief Array of step scales for optimization variables.
         */
        std::array<returnType, sizeof...(argType)> step_scales;
        /**
         * @brief Tuple representing the derivatives of the objective function.
         */
        std::tuple<argType...> derivatives {};
        /**
         * @brief Tuple representing the high values of the derivatives.
         */
        std::tuple<argType...> derivative_high {};
        /**
         * @brief  Unique pointer to the constraint manager. (Polymorphic class object)
         */
        std::unique_ptr<typename aux::constraints_system<returnType, argType...>::constraint_manager_base> constraint_manager_;
        /**
         * @brief Flag indicating if it's the first iteration settings.
         */
        bool first_iteration_settings = true;
        /**
         * @brief Flag indicating if constraints are enabled.
         */
        bool constraints_on = false;
        /**
         * @brief Flag indicating if classic gradient descent algorithm is used.
         */
        bool use_classic_gd = false;
        /**
         * @brief Flag indicating if derivative-based scaling is used.
         */
        bool use_scaling = false;
        /**
         * @brief Number of times the objective function is called.
         */
        std::size_t func_call_count = 0;


        template<class tupleType>
        returnType eval_func_at (tupleType&& IN_ARGS) noexcept {
            this->func_call_count++;
            if (this->constraints_on) {
                this->constraint_manager_->get_penalty(std::forward<tupleType>(IN_ARGS));
                return this->function->eval_func_at(std::forward<tupleType>(IN_ARGS)) + this->constraint_manager_->penalty;
            }
            return this->function->eval_func_at(std::forward<tupleType>(IN_ARGS));
        }

        void step_forward_with_secant_method (std::tuple<argType...> IN_POINT) noexcept {
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

        void step_forward_with_back_tracking (std::tuple<argType...> IN_POINT) {
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
