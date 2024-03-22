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
Gradient Decent algorithms are usually lacking of proper scalling when findind then next optimal point (See equation below). The learning rate ($\alpha$) is not proerly scalled to convert the differential set to the original equation set. Most Gradient Decent algorithms solve this issue by taking a constant learning rate or reduce by a factor the rate every iteration assuming a constant learning rate at the first iteration. In both cases, they are assumed. To, solve this, I have implmented a $secant method$ based scalling algorithm which aims to find the scale of the learning rate by relating the differential set to the original equation set.

```math
\begin{align}
\text{forward stepping: } X_{new} = X + \alpha \nabla f
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

## Secant Method Scalling
**Fact 1:** 
As the gradient decent progress, there will come a point where the step taken eventually skips past the optimal value (assuming large enough step is provided). In such cases, most algorithms use back-tracking to reduce the step to find the next best point which will reduce the $f$ value.

**Fact 2:** 
When Fact 1 is met, there exists another point, in the direction of current gradient, which is equal to current optimal value.

When both facts are met, we can use secant method to find the point mentioned in Fact 2. This will give us an accurate scale of $\alpha$ relating derivative set and original equation set. Now, with the scalling know, we can assume that the minima might be located at the $\alpha / 2$ position.

This allows for proper scalling of the learning rate and provides the perfect guess of learning rate at every iteration whenever the learning rate exceeds the forward stepping condition. Doing this 



