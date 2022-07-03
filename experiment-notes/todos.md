- [x] read NVAE paper
	- Residual Gaussian dist parameterization
	- KL balancing coeffs
	- KL warm-up
	- Spectral reg ?
	- How BU and TD nets are treated ?
- [ ] Read VACA paper https://arxiv.org/pdf/2110.14690.pdf
- [ ] Check BUnet activations, check sigmas
- [ ] Implement CS VAE Latent layer using GNN ? Is this a good idea? 
https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/06-graph-neural-networks.html

Thinking about a dataset (VACA can be helpful here)
Thinking about CNN information flow
Spatial vs. Conceptual latents / Micro vs Macro latents

Relating Graph Neural Networks to Structural Causal Models https://arxiv.org/abs/2109.04173


Disentanglement / Factorization pressure in GNN based priors / posteriors ?

Soundess of whole approach + architecture needs to be verified / re-thought ?



vae-disentanglement/train-logs/GNN_CS_VAE_dsprites_correlated_structure_test/version_0/checkpoints/epoch=9-step=115200.ckpt

I need to re-think how I visualize mu and logvae in the scalar metrics view. I think that their tendency to go towards zero is not an abberation. It is a result of me averaging the components over the batch . Since we reg towards N(0,1) it is natural for that average to be near 0. Histogram is a more accurate representation. Need to dwell on it and figure out a more insightful way of visualizing it.

LogVar_p_i has same plots for all i i.e. across all nodes
Mu_p_i has same plots for all i i.e. across all nodes

Run GNN based CSVAE with small number of node feature dim to check the dying dims phenomenon



