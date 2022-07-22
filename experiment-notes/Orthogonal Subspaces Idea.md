For all $z_i$ the associated features belong to a manifold in $\mathbb{R}^k$ . If we have $n$ nodes we want $n$ subspaces that are orthogonal to each other (pairwise orth.) in $\mathbb{R}^k$ 

This however seems to contradict the dependent nature of GNNs. Since for nodes that have parents the features are calculated as a function of features of parent nodes. How do we square this with this orth. subspaces idea ?

Secondly, afaik in 2D we can have a max of 2 orthogonal subspaces (here, 2 1D subspaces aka lines), any additional subspace will be non-orthogonal to at least one of them. So if we have $k$ nodes we need the feature dimension to be at least $\mathbb{R}^{k+1}$ for this idea to work? In 3D we have 3 1D subspaces

#### Orth subspaces may not be the solution
Orth. subspaces may not directly co-incide with the concept of disentanglement. To see why consider the following:

Let's assume that we have feat. dim of 3 which gives us 3 2d planes pairwise orth to each other. Namely the $xy, xz, yz$ planes.  

Consider a concept e.g. Hair color encoded onto $xz$ plane
And another concept hair length encoded onto $yz$.

Assume we get the following repr. for some example:
$$
\mu_1 = \begin{bmatrix} 
0.8 \\ 
0.3 \\
0.7 
\end{bmatrix}$$
Furthermore, in order to make sure that we only traverse a fixed set of dims we define a selector vector $\delta_D$ where the set $D$ gives us the index for dims that will be traversed. We will have $1$ in the corresponding dims and $0$ in others. In addition we have a traverse direction $\epsilon$ 
We traverse the concept "Hair color" as follows:

$\mu_1 + \delta_D \odot \ \epsilon = \begin{bmatrix} 0.8 \\ 0.3 \\ 0.7 \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} \odot \epsilon$ where $D = \{1,3\}$ for $xz$ plane.

Similarly for the concept "Hair length":

$\mu_1 + \delta_D \odot \ \epsilon  = \begin{bmatrix} 0.8 \\ 0.3 \\ 0.7 \end{bmatrix} + \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix} \odot \epsilon$ where $D = \{2,3\}$ for $yz$ plane.

But there is an immediate problem. In both cases we are changing the $3$rd dimension. Third dim encodes the information about both Hair color and Hair length. Hence these two concepts are entangled in the 3rd dim.

#### Can we use this to our advantage ?
Could this "sharing" phenomenon be used to our advantage for correlated nodes?
Nodes that are correlated can be allowed to share a dimension... In doing so the correlation is elegently captured in a single dimension.

Uncorrelated nodes will use or have to be encoded onto disjoint dims.

Right now corrs are modelled by GNN interactions. Can we limit those messages / interactions to specific "shared" dims for correlated nodes ?

Full disent would require us to encode concepts onto disjoint set of dims. We don't have to use 1 dim per concept and could potentially use more dims if number of dims is greater than the number of concepts.
$$ \mathcal{S} = \bigcup_i^K S_i \;\;\;\; s.t. \;\; S_i \subset \mathcal{S} , \;\;\;  S_i \cap S_j = \varnothing \forall i \neq j$$
This K gives us the number of disjoint dims. At minimum we nee K = num of unique elems. E,g for K=4 nodes we need at least 4 disjoint subsets


