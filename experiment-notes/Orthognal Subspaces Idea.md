For all $z_i$ the associated features belong to a manifold in $\mathbb{R}^k$ . If we have $n$ nodes we want $n$ subspaces that are orthogonal to each other (pairwise orth.) in $\mathbb{R}^k$ 

This however seems to contradict the dependent nature of GNNs. Since for nodes that have parents the features are calculated as a function of features of parent nodes. How do we square this with this orth. subspaces idea ?

Afaik in 2D we can have a max of 2 orthogonal subspaces (here, 2 1D subspaces aka lines), any additional subspace will be non-orthogonal to at least one of them.
So if we have $k$ nodes we need the feature dimension to be at least $\mathbb{R}^{k+1}$ for this idea to work? In 3D we have 3 1D subspaces
