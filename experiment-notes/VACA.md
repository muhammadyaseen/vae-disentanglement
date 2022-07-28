
- Do they impose any structure on $p(X)$ ?
- Do they impose any structure on $p(Z)$ ?
- Do $Z$ and $X$ have the same dimension ?
- What role does GNN play in decoder/encoder?
- Is there a limitation to complexity of encoder / decoder because of GNN?
- Can they handle non-linear relationships between $X$ components?
- How do $Z$ latents relate to $X$ semantically, if at all ?
- How do $Z$ latents relate to $U$ semantically, if at all ?
- On which part of the architecture do interventions take place?
- Did they check that the nodes in decoder correspond to the concepts in the data? 
- How do interventions work?

GNN operates on nodes given by components of $X = \{x_1, \ldots, x_k\}$
There are features associated with each node. These features depend on the node itself and its neighbours (or further down the hierarchy?)
The structure is imposed between $X$ components i.e $(x_i, x_j)$

SCM $\mathcal{M}( p(\textbf{U}), \tilde{\textbf{F}} )$    
|X| = |U| - they have same dimension
The structure of $p(X)$ is known in the form of a DAG / Adj. Mat.

X's are computed / generated directly from other X's and local exogen var U's
$$\tilde{\textbf{F}} = \{ X_i := \tilde{f}_i( \textbf{X}_{pa(i)},U_i)\}$$ The exact parametric forms of SCM equations $\tilde{\textbf{F}}$ are not known

$X$ - endogenous observed vars
$U$ - exogenous unobserved vars and $p(\textbf{U}) = \prod_i p(U_i)$
$Z$ - latent variable associated with each node $i$ -  $p(\textbf{Z}) = \prod_i p(Z_i)$

$X, U, Z$ all have same dimension $d$. It seems that components of $U$ can be composed of more than one unit (but doesn't really matter since we don't observe them they might all well be absorbed in 1-unit, we do multiple units in beta-VAE because we are trying to impose independence structure) 


Variational Graph Auto-Encoders take as input an adj. mat. $A$. It is used by encoder and decoder both to enfore structure on:
- The posterior approximation $q_\phi(\textbf{Z}|\textbf{X},\textbf{A})$ by Encoder GNN
- The likelihood $p_\theta(\textbf{X}|\textbf{Z},\textbf{A})$ by decoder GNN. Furthermore, $p_\theta(\textbf{X}|\textbf{Z},\textbf{A}) = \prod_i p_\theta(\textbf{X}_i|\textbf{Z},\textbf{A})$

such that it detemines which variables $X_i$ influence $Z_j \;\;\forall i,j \in [d]$. Does this mean $Z$ and $X$ have the same dimension $d$ ?

In contrast to this I want to impose the structure only on the latents such that the structure determines which nodes $Z_i$ influence $Z_j$. The whole of $X$ can influence any component of $Z$

"We summarize the main properties of an SCM that will allow us to propose a novel class of VGAEs, namely VACA, to compute accurate estimates of Observational, Interventional and Counterfactual distributions using observational data and a known causal graph.."

So VACA is a subclass of VGAEs ?
What do they mean by "properties of an SCM" ? Is this an assumption that is required in order for VACA to work? or these properties merely support the framework and are not a restriction upon the class of SCMs? (it is the latter)

3 Properties of SCM
Prop 1 - 
Prop 2 - 
Prop 3 - 

Necessary conditions on the design of encoder and decoder GNNs

Design Condition 1: Decoder GNN has at least $(\gamma - 1)$ hidden layes. $\gamma$ = longest directed path in the causal graph
Design Condition 2: Encoder GNN has no hidden layer

Two Propositions:


Relation between $Z$ and $U$

- Latent vars $Z$ play a similar role to exo. vars $U$
- Decoder $p_\theta(\textbf{X}|\textbf{Z},\textbf{A})$  plays a similar role to SE $\textbf{F}$. Both take a set of vars as input and produce $X$. At the same time Decoder does not aim to approximate the causal structural equations.
- $Z$ don't need to correspond to the true exogenous variables i.e. $p(\textbf{U}) \neq p(\textbf{Z})$ 
- They require that there is one independent latent variable $Z_i$ for every observed variable $X_i$ capturing all the information of $X_i$ that cannot be explained by its parents. Since by definition $X_i := f_i(\text{pa}(i), U_i)$, it means that $Z_i$ captures information contributed by $U_i$.
- Similar to the true posterior $p(U_i|X_i,\text{pa}(i))$, the distribution $p(Z_i|X_i,\text{pa}(i))$ should only depend on $X_i$ and its parents. (Shouldn't it be the variational distribution in 2nd part?)










