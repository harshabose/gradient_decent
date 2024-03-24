# A New Gradient Decent Algorithm
A New Gradient Decent Algorithm with Secant Method Scaling and Finite Difference Derivatives. Faster, Smarter and Versatile


Consider the following:
```math
\begin{align}
f:\;\;\mathbb{R}^d \longrightarrow \mathbb{R}
\end{align}
```

```math
\begin{align}
\text{subject to } Ax \leq B, \quad l_{i} \leq x_{i} \leq b_{i}
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

Gradient Descent algorithms often lack proper scaling when determining the next optimal point (See equation below). The learning rate ($\alpha$) is not properly scaled to convert the derivatives set into the objective variables set. Many Gradient Descent algorithms attempt to address this issue by using a constant learning rate and reducing it by a factor after each iteration, or by employing back-tracking algorithms. However, these approaches often result in poor convergence and consequently, a high iteration count. Users are burdened with the arduous responsibility of providing the optimal learning rate to strike a balance between a sufficiently large step and proper convergence. Predicting this learning step adds complexity. To alleviate this burden from the user, I have implemented a `Secant Method based Scaling algorithm`, which aims to find the learning rate by relating the differential set to the objective variables set.

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
- **Fact 1:**  As the gradient descent progresses, there comes a point where the step taken eventually overshoots the optimal value (assuming a large enough step is provided). In such cases, most algorithms use back-tracking to reduce the step size to find the next best point to find the next best value which reduces the objective function $f$.

- **Fact 2:**  When Fact 1 is met, there exists another point (let's call this the mirror point) in the direction of the current gradient that is equal to the current optimal value.

When both facts are met, we can use the secant method to find the mirror point. This will give us an accurate scale of $\alpha$, relating the derivative set to the objective variables set. Now, with the scaling known, we can assume that the minimum might be located at a position $\left(\alpha / 2\right)$ in the direction of the derivative vector. A mathematical representation is provided below:

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
\text{where } \alpha_{1} = 0 \text{ is the first root}
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

This enables proper scaling of the learning rate to map the derivative set onto the original equation set and provide an optimal estimate of the learning rate at each iteration whenever the learning rate exceeds the forward stepping condition. A pictographical representation is given below. Implementing this novel approach will significantly improve the convergence rate. See the example given below. <br><br>

<p align="center">
  <img src="https://github.com/harshabose/gradient_decent/assets/127072856/f96d3723-e8c4-43d6-adf1-a46c98393758" alt="Secant Scalling Method Image">
</p>

## What are included?
The gradient_decent class:
- `Detailed Doxygen Documentation:` has been provided for all classes, structs, and other utilities and assets. You can hover over any method to view the documentation.
- `Finite Difference:` was employed to compute the derivative, enabling optimisation of non-pure-mathematical "equations".
- `Secant Method scaling:` can be disabled, and a classic back-tracking Gradient Descent methodology could be employed instead.
- `Derivativee Scaling of learning rate:` employs the square root of the ratio of current derivatives to the highest derivatives encountered. This approach gradually reduces the learning rate as iterations progress. Even without secant method scaling, this enables a natural decline in the learning rate. Additionally, this feature can be toggled on or off based on user preference.
- `Momentum based derivative rotation:` introduces a mechanism where the current derivative vector at a point is rotated towards the weighted average of the previous derivative vectors, leveraging the heavy-ball gradient descent approach. This feature can be toggled on or off based on user preference. (\**TODO)
- `Supports for multi-dimensional:` is provided, allowing for the optimisation of functions with any number of dimensions. This flexibility enables the algorithm to handle a wide range of optimisation scenarios, accommodating diverse problem spaces.
- `Support for Constraints and bounds:` are also provided, utilising a linear penalty function with a user-defined slope and bounds projection to constrain the minimisation operation. This feature allows users to impose constraints on the optimisation process, ensuring that the solution adheres to specified conditions or limitations.
- `Effecient Coding Paradigms:` were employed to optimise the code, aiming to minimise memory usage and performance overhead. This approach ensures that the implementation is streamlined and resource-efficient, leading to faster execution and reduced computational costs.

## How to use?
1. Clone the repo at your desired location.
```bash
git clone https://github.com/harshabose/gradient_decent.git
```
2. This is a header-only file, just include the file in you code as shown below:
```cpp
#include "gradient_decent.h"
```
3. Create an instance of the class gradient_decent as shown below and set the parameters as desired and run:
```cpp
auto gradient_operator = gd::gradient_decent<double, double, double>(bivarient_function, 1.6, -1.2);
gradient_operator.perform_gradient_decent();
```
`NOTE: Requires C++20`

## Where can this be useful?
Gradient Descent Algorithms are used in various fields of science and engineering. My modifications can help these fields that need the flexibility of gradient descent but also can benefit from improved convergence rates and reduced computational burden. This modified algorithm can find a comfortable seat in various fields, including Machine Learning and Deep Learning, Numerical Optimisation and Scientific Computation, Image Processing and Computer Vision, Natural Language Processing (NLP), and more. Please give it a try and see if this modification can help with your use case.

## Example
The following code uses a bechmark 3D exponential objective function which has two minima at locations ${\sqrt{2}, -\sqrt{2}}$ and ${-\sqrt{2}, \sqrt{2}}$ with a minimum value of "0". <br><br>

<p align="center">
  <img src="https://github.com/harshabose/gradient_decent/assets/127072856/e72895d8-c0c4-4a14-a3c8-c3d0d3d8da6b" alt="bivarient">
</p>

<br>

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
//        gradient_operator->toggle_classic_gradient_algo();  // uncomment this to use classic GD

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
As can be seen, the number of iterations required, even with a far initial guess, is just 4. In contrast, the classic Gradient Descent algorithm took 47 iterations under the same settings. This stark difference highlights the clear advantage of the `Secant Method Scaling algorithm`. A graphical representation comparing the performance of the classic gradient descent and my algorithm is provided below: <br><br>

<br>

<img height="300" width="1141" alt="GD" src="https://github.com/harshabose/gradient_decent/assets/127072856/47eac3f3-0962-417b-b3fa-a5e21b7c2be6"><br><br>


<div align="center">
  <table class="tg">
    <thead>
      <tr>
        <th class="tg-uca5">Parameters</th>
        <th class="tg-0lax">Classic GD</th>
        <th class="tg-0lax">My GD Algo</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="tg-0lax">Initial Point</td>
        <td class="tg-0lax">(1.6, -1.2}</td>
        <td class="tg-0lax"><span style="font-weight:400;font-style:normal;text-decoration:none">(1.6, -1.2}</span></td>
      </tr>
      <tr>
        <td class="tg-0lax">Tolerance</td>
        <td class="tg-0lax">1e-3</td>
        <td class="tg-0lax">1e-3</td>
      </tr>
      <tr>
        <td class="tg-0lax">Iterations</td>
        <td class="tg-0lax">47</td>
        <td class="tg-0lax">4</td>
      </tr>
      <tr>
        <td class="tg-0lax">Time (micro-sec)</td>
        <td class="tg-0lax">59</td>
        <td class="tg-0lax">5</td>
      </tr>
      <tr>
        <td class="tg-0lax"><span style="font-weight:400;font-style:normal;text-decoration:none">Function Calls</span></td>
        <td class="tg-0lax">343</td>
        <td class="tg-0lax">35</td>
      </tr>
      <tr>
        <td class="tg-0lax"><span style="font-weight:400;font-style:normal;text-decoration:none">Derivative Scaling?</span></td>
        <td class="tg-0lax">off</td>
        <td class="tg-0lax">off</td>
      </tr>
      <tr>
        <td class="tg-0lax"><span style="font-weight:400;font-style:normal;text-decoration:none">Momentum Rotation?</span></td>
        <td class="tg-0lax">off</td>
        <td class="tg-0lax">off</td>
      </tr>
    </tbody>
  </table>
</div>



## Contribution
Contributions are welcome and greatly appreciated. Whether you're fixing a bug, implementing a new feature, or improving documentation, your efforts contribute to making this project better for everyone. Before contributing, please take a moment to review the guidelines in the CONTRIBUTING.md file to ensure a smooth and effective collaboration process. Thank you for your support!
