 VAE can be viewed as the natural evolution of recent robust PCA models, capable of learning nonlinear manifolds of unknown dimension obscured by gross corruptions.

Probing the basic VAE model under a few simplifying assumptions of increasing complexity whereby closed-form integrations are (partially) possible.

Canonical form of the VAE harbors an innate agency for robust outlier removal in the context of learning inlier points constrained to a manifold of unknown dimension. 

### Key contributions:
1. When the decoder mean $\mu_x$ is restricted to an affine function of $z$ (i.e. $Wz + b$) we prove that the VAE model collapses to a form of robust PCA (RPCA) which is a recently celebrated technique for separating data into low-rank (low-dimensional) inlier and sparse outlier components.
2. Explanantion of 2 central but underappriciated roles of the VAE encoder variance $\Sigma_z$.
	1. It facilitates learning the correct inlier manifold dimension.
	2. Can help smooth out undesirable minima in the energy landscape of what would otherwise resemble a traditional deterministic AE (had $\Sigma_z$ been $0$)

These points can have profound practical consequences, for ex:
Even if the decoder capacity is not sufficient to capture the gen dist within some fixed, unknown manifold, the VAE can nonetheless still often find the correct manifold itself, which is sufficient for deterministic recovery of uncorrupted inlier points. (TODO: elaborate this)

Sec 2: Two affine Decoder models and connections with probabilistic PCA approaches
Sec 3: Examine various partially affine Decoder models where only mean $\mu_x$ is affine while variance $\Sigma_x$ can have unlimited complexity and the encoder mean and var ($\mu_z$ and $\Sigma_z$) are also unconstrained.
Sec 4: Degeneracies in full VAE model that can arise even with a trivially simple encoder and corresponding latent representation.

### Sec 2 : Affine Decoder and Probabilistic PCA

With affine encoder and decoder models, the resulting deterministic network will simply learn principal components like vanilla PCA, a well-known special case of the AE.

In this sec, they state and prove that for a Decoder with Affine mean and Fixed Diagonal variance and an Encoder with unrestricted mean and variance the VAE objective collapses to Prob. PCA.

Some insights:
The diagonalization of $\Sigma_z$ collapses the space of globally minimizing solutions to a subset of the original. With more sophisticated parameterizations this partitioning of the energy landscape into distinct basins-of-attraction could potentially introduce suboptimal local extrema.

Even if $W$ is overparameterized (has larger dim than necessary) there exists an inherent mechanism to prune superfluous columns to exactly zero, i.e., column-wise sparsity. And once columns of $W$ become sparse, the corresponding elements of $\mu_z$ can no longer influence the data fit. 

So ultimately, sparsity of $\mu_z$ in this context is an artifact of the diagonal $\Sigma_z$ assumption and the interaction of multiple VAE terms.

Both variants of the Affine Decoder model lead to reasonable probabilistic PCA-like objectives regardless of how overparameterized $\mu_z$ and $\Sigma_z$ happen to be.

### Sec 3: Partially Affine Decoder and Robust PCA

In this sec, $\mu_x$ is still Affine.
But $\mu_z, \Sigma_z, \Sigma_x$ can have unrestricted parameterizations.

VAE is still able to self-regularize i.e global minimizers of VAE objective will correspond with optimal solutions to 
$$ \min_{L,S} \;\;\;\;\; n \times \text{rank}[L] + ||S||_0 \;\;\;\;\;  s.t. X = L + S $$
where $L=UV$ with $U, V$ low rank mats of appropriate dims and a sparse outlier component $S$. (An NP-hard discontinous opt. problem)


### Sec 4: Degeneracies Arising from a Flexible Decoder Mean

All $\mu_x, \mu_z, \Sigma_z, \Sigma_x$ can have unrestricted parameterizations.
It turns out that even if the latter three are severely constrained, overfitting will not be avoided when $\mu_x$ is over-parameterized. This is because the regulatory effects of $\Sigma_z$ can be completely squashed in these situations.

### The 3 hypotheses

1. When the decoder mean function is allowed to have multiple hidden layers of sensible size/depth, the VAE should behave like a nonlinear extension of RPCA and is likely to outperform it.

2. If the VAE latent representation $z$ is larger than needed then in the extended nonlinear case we would then expect that columns of the weight matrix $W$ from the first layer of the decoder mean network should be pushed to zero, effectively pruning away the impact of any superfluous elements of $z$. First decoder layer will have zero in columns corresponding to shut-off / useless dims.

3. When granted sufficient capacity in both $\mu_x ( \mu_z[x])$ and $\Sigma_x$ to model inliers and outliers respectively, the VAE should have a tendency to push elements of the encoder covariance $\Sigma_z$ to arbitrarily near zero along latent dimensions needed for representing inlier points. That is Variance of dims will be close to 1 for useless dims, and close to 0 for useful dims. This also means that if we have more dims in latent than necessary / than underlying latent manifold then we will have 1's corresponding to excess dims an 0's corresponding to useful / actual dims. 