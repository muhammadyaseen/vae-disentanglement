## Commonalities in DEAR and CausalVAE


| DEAR      | CausalVAE |
| ----------- | ----------- |
| $A$ - Binary adj. mat. provided for gen. tasks. Weights are learned      | $A$  - Structure is learned in "pre-train" stage via DAG constraint    |
| $X$   | $X$        |
| $Z$   | $Z$        |
| $\epsilon$ - Ind. exogen. vars used in Prior  | $\epsilon$ - Ind. exogen. vars used in Prior      |
| $\xi$ - True underlying factors  |   Not considered?     |
| $y$ - Annotated label of underlying factor s.t.  $\xi_i = \mathbb{E}(y_i \vert X)$  |  $u$ - Label associated with latent $z_i$ and $u_i = \mathbb{E}(z_i \vert X)$ since $z_i \sim \mathcal{N}(z_i ;  u_i, 1)$ |
| Composite Prior where $m$ latents are modelled ind. and $k$ have structure specified by $A$   |  $p_\theta(\epsilon, z \vert u)  = \mathcal{N}(0,I) \times \prod_i^n \mathcal{N}(z_i ;  u_i, 1)$ |
|  $Z = f_2(\;(I - A^T)^{-1}\;f_1(\epsilon)\;) := F_\beta(\epsilon)$ where $f_1$ is Identity and $f_2$ can be (invertable) linear and non-linear map| Linear SCM $\textbf{z} = \textbf{A}^T \textbf{z} + \epsilon = (I - \textbf{A}^T)^{-1} \epsilon  \;\;\;\; \epsilon \sim \mathcal{N}(0,I)$ and $z_i = g_i(A_i \odot z; \eta_i) + \epsilon_i$. Couldn't find how they are instantiating $g_i$
|They use a discriminator to estimate grad of KL | ELBO =  $\mathbb{E}_{q_\mathcal{X}}[\mathbb{E}_{q_\phi(z \vert x,u)} [\log p_\theta(x \vert z)]$ $- \mathcal{D}(q_\phi(\epsilon \vert x, u) \vert \vert p(\epsilon))$ $- \mathcal{D}(q_\phi(z\vert x, u) \vert \vert p_\theta(z \vert u))$ |
| Datasets: Pendulum, Flow, CelebA-4  |Datasets:  Pendulum, Flow, CelebA-4    |



- Why did both papers descrive only 4 concept graphs ? What happens to other latents, specially in CelebA tasks?
- Given the setting and experiments of these two works, what interesting contribution can I make?
- scrutinize my use of gnn and msgs

