# A New Gradient Decent Algorithm
Gradient Decent with Secant Method scaling. Faster, Robust and Smarter


Consider the following:
```math
\begin{align}
f:\;\;X^d \longrightarrow \mathbb{R}
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
Gradient Decent algorithms are usually considered lacking of proper scalling when findind then next point (See equation below). The learning rate $\alpha$ is not proerly scalled to convert the differential set to the original equation set.

```math
\begin{align}
X_{new} = X + \alpha \nabla f
\end{align}
```


