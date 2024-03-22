# A New Gradient Decent Algorithm
Gradient Decent with Secant Method scaling. Faster, Robust and Smarter


Consider the following:
```math
\begin{align}
f:\;\;\mathbb{R}^d \longrightarrow \mathbb{R}
\end{align}
```

```math
\begin{align}
\text{subject to } Ax \leq B, \quad l \leq x \leq b
\end{align}
```

```math
\begin{align}
\text{where } A = [ \,a_{i,\,j}\,] \in \mathbb{R}^{m \times n}
\end{align}
```

```math
\begin{align}
\min(f)
\end{align}
```
Gradient Decent algorithms are usually lacking of proper scaling when findind then next optimal point (See equation below). The learning rate ($\alpha$) is not proerly scalled to convert the derivatives set to the original equation set. Most Gradient Decent algorithms solve this issue by taking a constant learning rate and reduce by a factor the rate every iteration. This leads to excessive or limited forward. The user is given the arduous task of providing the nominal learning rate. Predicting this learning step has its own complexity. To alleviate this responsibility from the user, I have implmented a `Secant Method` based scaling algorithm which aims to find the the learning rate by relating the differential set to the original equation set.

```math
\begin{align}
\text{forward stepping: }
\end{align}
```
```math
\begin{align}
X_{new} = X + \alpha \nabla f
\end{align}
```
```math
\begin{align}
X_{new}\,, \, X \in \mathbb{R}^d \quad \nabla f \in \mathbb{R}
\end{align}
```
```math
\begin{align}
\text{subject to } f(X_{new}) < f(X)
\end{align}
```
```math
\begin{align}
\alpha \in \: ??
\end{align}
```

### Secant Method Scaling
- **Fact 1:**  As the gradient decent progress, there will come a point where the step taken eventually skips past the optimal value (assuming large enough step is provided). In such cases, most algorithms use back-tracking to reduce the step to find the next best point which will reduce the $f$ value.

- **Fact 2:**  When Fact 1 is met, there exists another point (lets call this mirror point), in the direction of current gradient, which is equal to current optimal value.

When both facts are met, we can use secant method to find the mirror. This will give us an accurate scale of $\alpha$ relating derivative set and original equation set. Now, with the scaling known, we can assume that the minima might be located at the $\left(\alpha / 2\right)$ position in the direction of derivative vector. A mathematical representation is given below:
```math
\begin{align}
g(\alpha) = f(X + \alpha \nabla f) - f(X)
\end{align}
```
```math
\text{where X is the current point and }X + \alpha \nabla f \text{ is the new point}
```
```math
\begin{align}
g(\alpha_{1}) = f(X + \alpha_{1} \nabla f) - f(X) = 0
\end{align}
```
```math
\text{where } \alpha_{1} \text{ is the current learning rate}
```
```math
\text{find the second root } (\alpha_{2}) \text{ of } g(\alpha) \text{ using Secant Method}
```
```math
X + \alpha_{2} \nabla f \text{ is the mirror point}
```
```math
\text{perfect guess } X_{new} = X + \left(\frac{\alpha_{2}}{2}\right) \nabla f
```

This allows for proper scaling of the learning rate to reflect the derivative set onto original equation set and provide the perfect guess of learning rate at every iteration whenever the learning rate exceeds the forward stepping condition. Doing this will improve the convergence rate by many folds. See the example below.

## What are included?
The gradient_decent class:
- `Finite Difference:` was used to calculate the derivative, allowing for optimation of non-pure-mathematical "equations".
- `Secant Method scaling:` can be turned off and a classic back-tracking methodology could be used instead.
- `Learning Rate Scaling:` using square root of ratio of current derivatives and the highest derivatives encountered (based on RMSprop) is used to reduce the learning as the iterations progress. This allows gradual decrease of learning rate even without secant method scaling allowing for natural decling of learning rate. This could also be turned off.
- `Momentum based derivative rotation:` is also introduced, where the current derivative vector at a point are rotated towards the weighted-average of the previous derivative vectors (based on heavy-ball gradient decent approach). This could also be turned off. (\** to be added)
- `Supports for multi-dimensional:` problems is given to support any number of dimensions.
- `Support for Constraints:` is also given which use a linear penalty function with a user defined slope to constraint the minimising operation.
- `Effecient Coding Paradigms:` were used to optimise the code to produce minimal memory and performance overhead. 

## How to use?
1. Clone the repo at your desired location.
```bash
git clone https://github.com/harshabose/gradient_decent.git
```
2. This is a header-only file, just include in you code as below:
```cpp
#include "gradient_decent.h"
```
3. Create an instance of the class gradient_decent as below and set the parameters as desired and run:
```cpp
auto gradient_operator = gd::gradient_decent<double, double, double>(bivarient_function, 1.6, -1.2);
gradient_operator.perform_gradient_decent();
```
**NOTE:** Requires C++20
## Example
The following code uses a bechmark 3D exponential equation and has one minima at locations ${\sqrt{2}, -\sqrt{2}}$ and with a value of "0". 

```cpp
#include <iostream>
#include <chrono>

#include "gradient_decent.h"


double bivarient_function (const double x, const double y) {
  constexpr double A = 10;
  return (A * x * y) / (std::exp(x * x + y * y)) + (5.0/std::exp(1.0));
}

int main () {
  std::tuple<double, double> lower_bounds = {-2.0F, -2.0F};
  std::tuple<double, double> upper_bounds = {2.0F, 2.0F};

  std::unique_ptr<gd::gradient_decent<double, double, double>> gradient_operator;
  gradient_operator = std::make_unique<gd::gradient_decent<double, double, double>>(bivarient_function, 1.6, -1.2);

  gradient_operator->add_lower_bounds(lower_bounds);
  gradient_operator->add_upper_bounds(upper_bounds);
  gradient_operator->set_tolerance(1e-3);

  auto start = std::chrono::high_resolution_clock::now();
  gradient_operator->perform_gradient_decent();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
}
```
Output with no compile-time optimisation: 
```bash
iteration @0 with optimal val at 1.48774 with point at {1.6, -1.2}
iteration @1 with optimal val at 0.013213 with point at {0.695668, -0.649016}
iteration @2 with optimal val at 4.05672e-06 with point at {0.706601, -0.708027}
iteration @3 with optimal val at 8.75034e-07 with point at {0.706751, -0.706773}

GD CONVERGED with optimal point at: {0.706751, -0.706773}
with optimal value: 8.75034e-07
Time taken: 5 microseconds
```
The true optimal at {0.707106, -0.707106} and {-0.707106, 0.707106} with optimal value of 0.0 (figure below). As can be seen, the number of iterations taken (even with a far initial guess) is 4. Compared to classic Gradient Decent which tool 300+ iterations for the same settings. This shows the clear advantage of the `Secant Method Scaling` technique. A pictographical representation of a classic gradient decent and my algorithm is given below:

<p align="center">
  <img width="460" height="300" src="https://github.com/harshabose/gradient_decent/assets/127072856/299edffc-cba9-4988-8a0f-4b4397cb6b07" alt="bivarient">
</p>

<img width="1141" alt="GD" src="https://github.com/harshabose/gradient_decent/assets/127072856/47eac3f3-0962-417b-b3fa-a5e21b7c2be6">


| Parameters     | My Algorithm | Classic GD |
| -------------- | ------------ | ---------- |
| Initial Point  | {1.6, -1.2}  | {1.6, -1.2}|
| Iterations     | 4            | 47         |
| Time           | 5            | 59         |
| Function Calls | 35           | 342        |
| Derivative Scalling ?     | off          | off        |
|tolerance       | 1e-3         | 1e-3       |


## Contribution
Contributions are welcome and greatly appreciated. Whether you're fixing a bug, implementing a new feature, or improving documentation, your efforts contribute to making this project better for everyone. Before contributing, please take a moment to review the guidelines in the CONTRIBUTING.md file to ensure a smooth and effective collaboration process. Thank you for your support!
