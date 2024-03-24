/**
 * @file gradient_decent.h
 * @brief Header file defining the gradient_decent and function wrapper classes for new GD approach using
 * secant method scaling of learning rate.
 *
 * @author Harshavardhan Karnati
 * @date 07/03/2024
 */

#ifndef CONCEPTUAL_GRADIENT_DECENT_H
#define CONCEPTUAL_GRADIENT_DECENT_H


/**
 * @brief Defines macros for conditional verbose printing.
 *
 * This set of macros defines conditional verbose printing functionality based on the value of the VERBOSITY
 * preprocessor macro. It allows printing debug messages to the standard output stream when VERBOSITY is enabled.
 *
 * @note
 * - The macros include VERBOSE_PRINT, VERBOSE_PRINT_, and _VERBOSE_PRINT_, which print messages with or without
 *   line breaks based on the VERBOSITY setting.
 * - When VERBOSITY is enabled (VERBOSITY = 1), debug messages are printed to the standard output stream.
 * - When VERBOSITY is disabled (VERBOSITY = 0), the macros expand to empty statements, avoiding any overhead
 *   in production code.
 */
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
#include "meta_types.h"

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
         * Constructs a gradient descent optimiser with NSDMI default parameters.
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
         * auto constraint1_ = aux::constraints_system<returnType, argsType...>::create_constraint<decltype(constraint_function_1), double>(constraint_function_1, "&#x3C;", 9.0, 0.001f);
         * @endcode
         *
         * @note Constraints must be added before performing the optimisation.
         * The method assumes that the provided constraint objects have member variables:
         * `func`, `value`, `operator_`, and `tolerance`.
         * */
        template <template <class, class> class... createConstraintType, class constraintFuncType, typename valueType>
        void add_constraints(createConstraintType<constraintFuncType, valueType>&... constraints) noexcept {
            this->constraints_on = true;
            this->constraint_manager_ = std::make_unique<typename aux::constraints_system<returnType, argType...>::template constraint_manager<decltype(constraints.func)...>>(std::move(constraints.func)..., (constraints.value)...);
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

        /**
         * @brief Evaluates the objective function at the specified arguments.
         *
         * This method evaluates the objective function at the specified arguments provided as a tuple.
         * It increments the function call count and, if constraints are enabled, adjusts the objective
         * function value based on the penalty imposed by the constraint manager.
         *
         * @tparam tupleType The type of the tuple containing arguments.
         * @param IN_ARGS The tuple containing arguments at which the function is evaluated.
         * @return The result of evaluating the objective function at the specified arguments.
         *
         * @details
         * The eval_func_at method increments the function call count to keep track of the number of times
         * the objective function is called during optimisation. If constraints are enabled, it retrieves
         * the penalty from the constraint manager based on the provided arguments and adds it to the
         * objective function value. The adjusted value is then returned as the result. If constraints are
         * not enabled, the objective function is evaluated without any adjustments and the result is returned.
         *
         * @note
         * <ul>
         * <li> This method is noexcept, ensuring that it does not throw exceptions allowing for compile-time optimisation
         * <li> The provided arguments are forwarded to the objective function for evaluation.
         * </ul>
         */
        template<class tupleType>
        returnType eval_func_at (tupleType&& IN_ARGS) noexcept {
            this->func_call_count++;
            if (this->constraints_on) {
                this->constraint_manager_->get_penalty(std::forward<tupleType>(IN_ARGS));
                return this->function->eval_func_at(std::forward<tupleType>(IN_ARGS)) + this->constraint_manager_->penalty;
            }
            return this->function->eval_func_at(std::forward<tupleType>(IN_ARGS));
        }

        /**
         * @brief Performs a step forward using the Secant Method in the gradient descent algorithm.
         *
         * This method implements a step forward using the Secant Method in the gradient descent algorithm.
         * It adjusts the optimal point based on the next point created with bounds projection, evaluates
         * the objective function at the adjusted point, and updates the learning rate and optimal value
         * based on the Secant Method. If the updated optimal value is lower than the current optimal value,
         * the current tolerance is updated, and the process continues.
         *
         * @param IN_POINT The current point at which the step forward is performed.
         *
         * @details
         * The step_forward_with_secant_method method performs a step forward in the optimisation process
         * using the Secant Method. It first creates the next point using bounds projection and evaluates
         * the objective function at that point. If the objective function value at the new point is greater
         * than the current optimal value, the learning rate is adjusted using the Secant Method, and the
         * process is repeated with the updated learning rate. If the updated optimal value is lower than
         * the current optimal value, the current tolerance is updated, and the process continues.
         *
         * @note
         * - This method is noexcept, ensuring that it does not throw exceptions.
         * - The Secant Method is used to dynamically adjust the learning rate during optimisation.
         * - The bounds_projection method adjusts the optimal point to ensure it falls within specified bounds.
         * - The eval_func_at method is used to evaluate the objective function at different points.
         */
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

        /**
         * @brief Performs a step forward using the backtracking algorithm in the gradient descent.
         *
         * This method implements a step forward using the backtracking algorithm in the gradient descent.
         * It repeatedly adjusts the optimal point using bounds projection, evaluates the objective function
         * at the adjusted point, and updates the learning rate based on backtracking. The process continues
         * until the objective function value decreases or a maximum number of iterations is reached.
         *
         * @param IN_POINT The current point at which the step forward is performed.
         *
         * @details
         * The step_forward_with_back_tracking method performs a step forward in the optimisation process
         * using the backtracking algorithm. It initializes iterative counters and maximum iteration limits
         * to control the backtracking process. It repeatedly adjusts the optimal point using bounds projection,
         * evaluates the objective function at the adjusted point, and checks if the objective function value
         * decreases. If the objective function value decreases, the current tolerance is updated, and the process
         * stops. If the maximum iteration limit is reached without finding a suitable point, an exception is thrown.
         *
         * @note
         * - This method throws a runtime_error if it cannot find the next point using the backtracking algorithm.
         * - The backtracking algorithm adjusts the learning rate to ensure convergence towards the optimal point.
         * - The bounds_projection method adjusts the optimal point to ensure it falls within specified bounds.
         * - The eval_func_at method is used to evaluate the objective function at different points.
         */
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

        /**
         * @brief Helper function to set the highest derivatives for a specific index.
         *
         * This method is a helper function used by the set_high_derivatives method to set the highest derivatives
         * for a specific index in the tuple of derivatives. It compares the absolute value of the derivative at
         * index 'i' with the absolute value of the corresponding derivative in the derivative_high tuple. If the
         * absolute value of the derivative at index 'i' is greater than the absolute value of the corresponding
         * derivative in the derivative_high tuple, it updates the learning rate to 1.0 and updates the corresponding
         * derivative in the derivative_high tuple.
         *
         * @tparam i The index of the derivative.
         */
        template <std::size_t i>
        void set_high_derivatives_helper () {
            if (std::abs(std::get<i>(this->derivatives)) > std::abs(std::get<i>(this->derivative_high))) {
                this->learning_rate = 1.0;
                std::get<i>(this->derivative_high) = std::get<i>(this->derivatives);
            }
        }

        /**
         * @brief Sets the highest derivatives for all indices specified in the index sequence.
         *
         * This method sets the highest derivatives for all indices specified in the index sequence 'i_seq'.
         * It calls the set_high_derivatives_helper method for each index in the sequence.
         *
         * @tparam i The indices for which the highest derivatives are set.
         * @param i_seq The index sequence specifying the indices for which the highest derivatives are set.
         */
        template <std::size_t... i>
        void set_high_derivatives (std::index_sequence<i...>) {
            (this->set_high_derivatives_helper<i>(),...);
        }

        /**
         * @brief Helper function to calculate derivatives at specified indices.
         *
         * This method calculates the derivatives at the specified indices for the given tuple of arguments.
         * It uses a finite difference method to approximate the derivatives. If an exception occurs during
         * calculation, it falls back to using the backward finite difference method. The calculated derivatives
         * are then scaled if derivative scaling is enabled.
         *
         * @tparam tupleType The type of the tuple containing the arguments.
         * @tparam i The indices at which derivatives are calculated.
         * @param IN_TUPLE The tuple of arguments.
         * @return A tuple containing the calculated derivatives at the specified indices.
         */
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

        /**
         * @brief Calculates derivatives at the specified point.
         *
         * This method calculates the derivatives at the specified point using the finite difference method.
         * It then sets the highest derivatives and scales the derivatives if derivative scaling is enabled.
         *
         * @tparam tupleType The type of the tuple containing the point coordinates.
         * @param IN_POINT The point at which derivatives are calculated.
         * @return A tuple containing the calculated derivatives.
         */
        template<class tupleType>
        std::tuple<argType...> calculate_derivatives_at (tupleType&& IN_POINT) {
            this->derivatives = this->calculate_derivatives_at_helper(std::forward<tupleType>(IN_POINT), indices_for_args{});
            this->set_high_derivatives(indices_for_args{});
            if (this->use_scaling) this->scale(this->step_scales.data(), indices_for_args{});
            return this->derivatives;
        }

        /**
         * @brief Projects a point onto the bounds defined by lower and upper bounds.
         *
         * This method projects each coordinate of the input point onto the bounds defined by the lower
         * and upper bounds. If a coordinate is outside the bounds, it is replaced by the corresponding
         * bound. The projection is performed for each coordinate specified by the index sequence.
         *
         * @tparam i The indices of coordinates to project.
         * @param IN_POINT The input point to be projected.
         * @return The projected point after bounds projection.
         */
        template <std::size_t... i>
        std::tuple<argType...>&& bounds_projection (std::tuple<argType...>&& IN_POINT, std::index_sequence<i...>) noexcept {
            // Project each coordinate onto the bounds if IN_POINT is outside of bounds
            (((std::get<i>(IN_POINT) < std::get<i>(this->lower_bounds)) ? std::get<i>(IN_POINT) = std::get<i>(this->lower_bounds) : std::get<i>(IN_POINT) = std::get<i>(IN_POINT)),...);
            (((std::get<i>(IN_POINT) > std::get<i>(this->upper_bounds)) ? std::get<i>(IN_POINT) = std::get<i>(this->upper_bounds) : std::get<i>(IN_POINT) = std::get<i>(IN_POINT)),...);
            return std::forward<std::tuple<argType...>&&>(IN_POINT);
        }

        /**
         * @brief Checks if the current optimal point lies within the defined bounds.
         *
         * This method checks if each coordinate of the current optimal point lies within the bounds
         * defined by the lower and upper bounds. It returns true if all coordinates are within bounds,
         * and false otherwise. The check is performed for each coordinate specified by the index sequence.
         *
         * @tparam i The indices of coordinates to check.
         * @return True if the optimal point is within bounds, false otherwise.
         */
        template <std::size_t... i>
        bool check_point_bounds (std::index_sequence<i...>) {
            // Check if each coordinate is within bounds
            return ((std::get<i>(this->optimal_point) < std::get<i>(this->lower_bounds)) && ...) &&
                   ((std::get<i>(this->optimal_point) > std::get<i>(this->upper_bounds)) && ...);
        }

        /**
         * @brief Prints the elements of a tuple to the verbose output stream.
         *
         * This method prints the elements of the specified tuple to the verbose output stream, enclosed
         * within curly braces. The printing is conditional based on the VERBOSITY flag. If VERBOSITY is
         * enabled, the elements are printed; otherwise, no action is taken.
         *
         * @tparam tupleType The type of the tuple containing elements to be printed.
         * @tparam i The indices of elements to print.
         * @param IN_TUPLE The tuple containing elements to be printed.
         */
        template<class tupleType, std::size_t... i>
        void verbose_print_tuple (tupleType&& IN_TUPLE, std::index_sequence<i...>) {
#if VERBOSITY
            _VERBOSE_PRINT_("{");   // Start of tuple printing
            // Lambda function to print elements of the tuple
            auto print_tuple_at = [] (tupleType&& IN_TUPLE_) -> void {
                // Print each element of the tuple, separated by commas
                ((std::cout << (i == 0? "" : ", ") << std::get<i>(IN_TUPLE_)),...);
            };
            // Call the lambda function to print elements of the tuple
            print_tuple_at(std::forward<tupleType>(IN_TUPLE));
            VERBOSE_PRINT("}");     // End of tuple printing
#endif
        }

        /**
         * @brief Calculates the Euclidean distance between two tuples.
         *
         * This method calculates the Euclidean distance between two tuples of the same type,
         * element-wise. The distance is computed as the square root of the sum of squared
         * differences between corresponding elements in the tuples.
         *
         * @tparam i Indices of elements in the tuples.
         * @param IN_FIRST_TUPLE The first tuple.
         * @param IN_SEC_TUPLE The second tuple.
         * @return The Euclidean distance between the two tuples.
         */
        template <std::size_t... i>
        auto get_distance_tuple (std::tuple<argType...> &IN_FIRST_TUPLE, std::tuple<argType...> &IN_SEC_TUPLE, std::index_sequence<i...>) {
            auto sum = (((std::get<i>(IN_FIRST_TUPLE) - std::get<i>(IN_SEC_TUPLE)) * (std::get<i>(IN_FIRST_TUPLE) - std::get<i>(IN_SEC_TUPLE))) + ...);
            return std::sqrt(sum);
        }

        /**
         * @brief Calculates the tolerance value.
         *
         * This method calculates the tolerance value by adding the current tolerance to
         * the Euclidean distance between the current optimal point and the old optimal point.
         *
         * @return The calculated tolerance value.
         */
        returnType get_tolerance () {
            returnType tol = this->current_tolerance + this->get_distance_tuple(this->optimal_point, this->old_optimal_point, indices_for_args{});
            return tol;
        }

        /**
         * @brief Creates the next point based on the current point and derivatives.
         *
         * This method calculates the coordinates of the next point based on the current point
         * and the derivatives. The new point is computed using the gradient descent update rule.
         *
         * @tparam i Indices of elements in the tuples.
         * @param IN_POINT The current point.
         * @return The next point.
         */
        template <std::size_t... i>
        std::tuple<argType...> create_next_point (const std::tuple<argType...> &IN_POINT, std::index_sequence<i...>) noexcept {
            return std::move(std::make_tuple((std::get<i>(IN_POINT) - std::get<i>(this->derivatives) * this->learning_rate * this->step_scales.at(i))...));
        };

        /**
         * @brief Scales a given value using a scaling factor.
         *
         * This function scales a given value using a scaling factor. It computes the square
         * root of the absolute value of the ratio of the input value to the scaling factor.
         *
         * @param IN_X The value to be scaled.
         * @param IN_FACTOR The scaling factor.
         * @return The scaled value.
         */
        returnType scale_function (const returnType IN_X, const returnType IN_FACTOR) {
            return std::sqrt(std::abs(IN_X / IN_FACTOR));
        }

        /**
         * @brief Scales an array of values based on derivative magnitudes.
         *
         * This method scales an array of values based on the magnitudes of derivatives. It
         * adjusts the scaling factor for each element of the array according to the derivative
         * magnitude and the highest derivative magnitude encountered so far.
         *
         * @tparam i Indices of elements in the array.
         * @param IN_TO_SCALE Pointer to the array to be scaled.
         */
        template <std::size_t... i>
        void scale (returnType* IN_TO_SCALE, std::index_sequence<i...>) {
            if (!(this->first_iteration_settings)) {
                ((*(IN_TO_SCALE + i) *= this->scale_function(std::get<i>(this->derivatives), std::get<i>(this->derivative_high))),...);
                ((*(IN_TO_SCALE + i) = *(IN_TO_SCALE + i) < this->tolerance ? this->tolerance : *(IN_TO_SCALE + i)),...);
            }
        }

        /**
         * @brief Computes the learning rate adjustment using the Secant Method.
         *
         * This method computes the adjustment of the learning rate using the Secant Method
         * to optimise the gradient descent algorithm. It finds the learning rate which relates the
         * derivative set space to the objective function variable set space. It iteratively calculates
         * the new learning rate based on the current and previous learning rates, and the corresponding
         * objective function values. The process continues until the convergence criterion is met or until
         * the maximum number of iterations is reached.
         *
         * @param IN_CURRENT_VAL The current value of the objective function.
         * @param IN_REQUIRED_VAL The required value of the objective function.
         * @return The adjusted learning rate computed using the Secant Method.
         *
         * @details
         * The secant_learning_rate_scaling method adjusts the learning rate used in the gradient descent
         * algorithm based on the Secant Method. It dynamically updates the learning rate to achieve convergence
         * to the optimal solution. The method iteratively computes the new learning rate by interpolating between
         * the current and previous learning rates, based on the corresponding objective function values. This process
         * continues until the difference between successive learning rates falls below a predefined threshold or until
         * the maximum number of iterations is reached.
         *
         * @note
         * <ul>
         * <li> This method is noexcept, ensuring that it does not throw exceptions.
         * <li> The Secant Method is a root-finding algorithm used to approximate roots of a real-valued function.
         * <li> The computed learning rate adjustment helps in optimising the gradient descent algorithm's convergence.
         * </ul>
         */
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
