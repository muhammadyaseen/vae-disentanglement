Goal: Propose a causal discovery method where variables $Z$ in SCM are observed with the aim of learning the causal structure among $Z$ i.e. the matrix $A$.

Linear SEM: $$Z = A^T Z + \epsilon \;\;\;\ \text{eq (1)}$$
Ancestral form of Lin. SEM: $$Z = (I - A^T)^{-1} \epsilon \;\;\;\ \text{eq (2)}$$Eq (2) can be written as: $Z = f_A(\epsilon)$ which is a general form recognized by the DL community as an abstraction of parameterized GNNs that take node features $\epsilon$ as input and return $Z$ as high level representations. Nearly all GNNs can be written in this form

Owing to special structure of (2) we propose a new GNN arch. 
$$ Z = f_2(\;(I - A^T)^{-1}\;f_1(\epsilon)\;) \;\;\;\ \text{eq (3)}$$
Functions $f_1$ and $f_2$ effectively perform (possibly nonlinear) transforms on $\epsilon$ and $Z$, respectively.

If $f_2$ is invertible then (3) is equivalent to:
$$ f_2^{-1}(Z) = A^T f_2^{-1}(Z) + f_1(\epsilon) $$
Let $$ \hat Z = f_2^{-1}(Z) \text{ and } \hat \epsilon = f_1(\epsilon)$$
Then it becomes: $$ \hat Z = A^T \hat Z + \hat \epsilon $$
which is a generalized version of the Linear SEM (1)

Proposed corresponding encoder: $$\epsilon = f_4((I-A^T) f_3(Z))$$$f_3$ and $f_4$ are parameterized functions that conceptually play the inverse role of $f_1$ and $f_2$.
