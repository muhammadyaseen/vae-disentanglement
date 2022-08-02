- ~~bidir gen models~~
- indetifyability and consistency of DEAR
- causal controllable generation
- implicit distributions
- what is "causal ordering"
- why do they need invertibility for interventions?

DEAR: Disentangled gEnerative cAusal Representation

### Summary
- Unlike existing disentanglement methods that enforce independence of the latent variables, we consider the general case where the underlying factors of interests can be causally correlated. 
- Previous methods with independent priors fail to disentangle causally correlated  factors. (This has been proven theoretically as welll in '**On Representations Learned from Correlated Data**' paper by Locatello et. al.)
- The key  ingredient of this new formulation is to use a SCM as the prior for a bidirectional (??) generative model. The prior is then trained jointly with a generator and an encoder using a suitable GAN loss incorporated with supervision. 
- SCM Prior is a DAG structure over the latent variables / concepts. To implement it they use a prior GNN which uses the structure from the given / assumed SCM.

Section 3: Introduce the problem setting of disent. gen. causal Repr. learning and identify a problem with previous methods.

Section 4: Propose DEAR and provide theoretical justifications on both identifiability and asymptotic consistency.

Section 5: Empirical studies concerning causal controllable generation, downstream tasks and structure learning.

### Section 3: Problem setting
#### 3.1 - Gen Model
- Normal VAE setting 
- ELBO allows a closed form to be optimized easily only with factorized Gaussian prior, encoder and generator (Shen et al., 2020)
- Implicit distributions, where the randomness is fed into the input or intermediate layers of the network, are favored over factorized Gaussians in terms of expressiveness. Then minimizing ELBO requires adversarial training, as discussed in Sec 4.3.

#### 3.2 - Supervised regularizer
A bit different / more formalized than usual
- $\xi \in \mathbb{R}^m$ is the dist of true underlying factors
- $y_i$ is an annotated observation of $i$-th factor. That is, $y_i = f_{ann}(\xi_i)$
- They require $\xi_i = \mathbb{E}[ y_i | X]$ - not sure how to visualize or understand it yet
- Supervision is applied to the deterministic part of Encoder i.e. the mean.
- Sup. loss is of the form:

$$
\begin{equation}
l_{sup,i} = 
	\begin{cases}
	CE( \mu(X)_i, y_i) & \text{if } y_i \text{ is binary or bounded label for } \xi_i  \\
	\\
	(\mu(X)_i - y_i)^2 & \text{if } y_i \text{ is continuous label for } \xi_i
	
	\end{cases}
\end{equation}
$$
Simply put, $i$-th dim of $\mu$ should have predictive power for $i$-th latent label.

Note that they only use a single unit to represent a single concept / latent. This is consistent with previous work. (I'll need a good reason to justify using multiple units, if I do it)

Locatella et. al. don't distinguish between the actual underlying latent and its observaiton. This seems largely a pedantic point, is there are actual reason for doing this?

#### 3.3 Unidentifiability with an independent prior

### Section 4: Causal Disent Learning
#### 4.1.1 SCM Prior
- Causal / SCM prior 
They adopt the general non-linear SCM proposed by Yue et al. 
(**DAG-GNN: DAG Structure Learning with Graph Neural Networks**) details in [[DAG-GNN]] note.

Specifically,
$$ Z = f_2(\;(I - A^T)^{-1}\;f_1(\epsilon)\;) := F_\beta(\epsilon)$$
Where:
- $\epsilon$ are the exogenuous vars and follow $\mathcal{N}(0,I)$
- $f_2$ and $f_1$ are element-wise transformations that are generally non-linear.
- $A$ is the **weighted** adj. mat. 
- $\beta = (f_2, f_1, A)$ denotes the parameters of elements in the paranthesis 
- $I_A$ is the binary adj. matrix.

When $f_2$ is invertible (a) is equivalent to:
 $$f_2^{-1}(Z) = A^T f_2^{-1}(Z) + f_1(\epsilon)$$
which indicates that the factors $Z$ satisfy a linear SCM after nonlinear transformation $f_2$, and enables interventions on latent variables.

#### Implementation of the SCM (details from Appendix)
- We find Gaussians are expressive enough as unexplained noises, so we set $f_1$ as the identity mapping.
- We require the invertibility of $f_2$, we implement both linear and non-linear instantiations
- For linear $f_2$ we have $f_2(z) = Wz + b$, where $W$ is a diagonal mat to model element-wise transformation
- For non-linear $f_2$ we use piece-wise linear functions


#### 4.1.2 Learning of $A$
Learning of $A$ matix: Assume super-graph of true binary graph is available.

#### 4.1.3 Generation from interventional distributions
They also model interventions over latent variables.


#### 4.1.4 Latent dim and Composite prior
If we have $m + k = M$ number of total underlying factors and we're only interesting to model the causal graph / interaction between $m$ variables and will model the other $k$ factors as independent then we can have a composite prior.
They propose to use a prior that is a composition of a causal model for the first $m$  dimensions  and another distribution for the other $k$ dimensions to capture other factors necessary for generation, like a standard Gaussian. 
In this way the first $m$ dims of $Z$ aim at learning the DR of the $m$ factors of interests, while the role of the remaining $k$ dims is to capture other factors that are necessary for generation but whose structure is neither cared nor explicitly modeled.

### 4.2 DEAR formulation and identifiability of disentanglement 
"We have one more module to learn which is the SCM prior." 

What is there to learn in the SCM prior? 
They already assume to know the structure... I think they're talking about learning the params of transformations $f_1$ and $f_2$ and the weights of mat $A$?

$$L_{gen} = D_{KL}( q_E(X,Z), p_{G,F}(X,Z)) $$
$$\min_{E,G,F} \;\; L(E,G,F) := L_{gen}(E,G,F) + \lambda L_{sup}(E)$$
### 4.3 Algorithm
- We use an implicit generated conditional $p_G(X|Z)$, where we inject Gaussian noises to each convolution layer in the same way as Shen et al. (2020).
- Then the SCM prior $p_F(Z)$ and implicit $p_G(X|Z)$ cause $L_{gen}$ to lose an analytic form. Hence we adopt a GAN method to adversarially estimate the gradient of $L_{gen}$ as in Shen et al. (2020). 
- Different from their setting, the <u>prior also involves learnable parameters</u>, that is, the parameters $\beta = (f_2, f_1, A)$ of the SCM.





Do they use an encoder GNN / How do they model the variational posterior ?

Does their var. posterior have or mirrors the SCM structure ?

In the paper there's no explanation of why the units learn the concepts they do. There's probably an implication that they do so because they're being supervised by respective labels... but is that really enough ?
