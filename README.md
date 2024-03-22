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
Gradient Decent algorithms are usually lacking of proper scalling when findind then next optimal point (See equation below). The learning rate ($\alpha$) is not proerly scalled to convert the differential set to the original equation set. Most Gradient Decent algorithms solve this issue by taking a constant learning rate or reduce by a factor the rate every iteration assuming a constant learning rate at the first iteration. In both cases, they are assumed. To, solve this, I have implmented a **Secant Method** based scalling algorithm which aims to find the scale of the learning rate by relating the differential set to the original equation set.

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
X_{new}, \, X \in \mathbb{R}^d \quad \nabla f \in \mathbb{R}
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

### Secant Method Scalling
- **Fact 1:**  As the gradient decent progress, there will come a point where the step taken eventually skips past the optimal value (assuming large enough step is provided). In such cases, most algorithms use back-tracking to reduce the step to find the next best point which will reduce the $f$ value.

- **Fact 2:**  When Fact 1 is met, there exists another point, in the direction of current gradient, which is equal to current optimal value.

When both facts are met, we can use secant method to find the point mentioned in Fact 2. This will give us an accurate scale of $\alpha$ relating derivative set and original equation set. Now, with the scalling know, we can assume that the minima might be located at the $\alpha / 2$ position.

This allows for proper scalling of the learning rate and provides the perfect guess of learning rate at every iteration whenever the learning rate exceeds the forward stepping condition. Doing this will improve the convergence rate by many folds. See the example below.

## What are included?
The gradient_decent class:
- **Secant Method scalling:** can be turned off and a classic back-tracking methodology could be used instead.
- **Learning Rate Scalling:** using sqrt ratio of current derivatives and the highest derivatives encountered (based on RMSprop) is provided. This allows gradual decrease of learning rate even without secant method scalling allowing for natural decling of learning rate. This could also be turned off.
- **Momentum based derivative rotation:** is also introduced, where the current derivative vector at a point are rotated towards the weighted-average of the previous derivative vectors (based on heavy-ball gradient decent approach). This could also be turned off.
- **Support for Constraints:** is also given which use a linear penalty function with a user defined slope (optional) to constraint the minimising operation.
- **Effecient Coding Paradigm:** such as tempalted lambda and meta-programming were used to optimise the code to produce minimal memory and performance overhead. 

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
## Example





