What can VAE do if two attributes / features seem to vary together in the images... keeping in mind the PCA interpretation and Information Theoretic interpretation...

In initial stages/epochs it has been seen that during training VAE reconstructions look like "mode" or "average" of the images in the datase..
1. a blurry circle in dsprites
2. a blurred face with both feminine and masculine features in celeba
3. a hazy pendulum in pendulum, and so on

The Understanding Paper took the approach to interpret the latent layer as noisy channels which transmit information about image $X$ via latents $z_i$
If we continue with that interpretation then in my case it will mean that we're working with cascaded channels.. channels that feed into each other in a strictly ordered hierarchy..

- [ ] I need to see to what extent the explanation in UdBetaVAE paper DEPENDS on the factors being independent and how can it be tweaked / modified for correlated case. What is the most reasonable thing that should happen in correlated case
- [ ] How can this cascaded channels idea help if at all?

The capacity of the latent channels can only be increased by (these two things both increase the KL term w.r.t N(0,I) prior):
1. Dispersing the posterior means across the data points, or
2. Decreasing the posterior variances (why can't an increase of variance increase KL? Inc var means more spread out / non-local dists)

Reconstructing under this bottleneck encourages embedding the data points on a set of representational axes where nearby points on the axes are also close in data space.

KL can be minimised by:
1. Reducing the spread of the posterior means, or 
2. Broadening the posterior variances, 

i.e. by squeezing the posterior distributions into a shared coding space.


We can think about this in terms of the degree of overlap between the posterior distributions across the dataset. The more they overlap, the broader the posterior distributions will be on average (relative to the coding space), and the smaller the KL divergence can be. 
However, a greater degree  of overlap between posterior distributions will tend to result in a cost in terms of log likelihood due to their reduced average discriminability.

Under a constraint of maximising such overlap (<span class="remark">what exactly is causing the constraint? I think they mean the constraint of minimizing KL</span>), the smallest cost in the log likelihood can be achieved by arranging nearby points in data space close together in the latent space.. So even if they overlap and the recon is ambiguous, it is the least possible ambiguity. This would result in very minor differences between original and recon and hence low recon error. For ex instead of recon at position x=20 it recons at x=19, or slightly different angle etc. <span class="remark">So the latent space exhibits this continum structure because of this contraint.. should understand it clearly</span>

Our key hypothesis is that β-VAE finds latent components which make different  
contributions to the log-likelihood term of the cost function. These latent components tend to correspond to features in the data that are intuitively qualitatively different, and therefore may align with the generative factors in the data. <span class="remark">Although the hypothesis is reasonable, the term 'different contributions to log-lkd cost func' is too vague. I need to think about it more critically considering my correlated setting</span>

Intuitively, when optimising a pixel-wise decoder log likelihood, information about position will result in the most gains compared to information about any of the other factors of variation in the data, since the likelihood will vanish if reconstructed position is off by just a few pixels. <span class="remark">This might explain the behaviour I have seen during training where the "average" position of objects in the image is captured first before any details start to emerge.. in celeba which is fairly complicated dataset the decoder / network manages to get the 'structure' of the face correct.. then fills in details later as training progresses</span>
<span class="remark">A corollary of this might also be that some factors just might not be worth (the increase in lkd) to encode because of the price (inc in KL) we have to pay.. so we might miss on some fine details. This will also be affected by how the terms have been weighted</span>
<span class="remark">Think how correlated features would be affected by this. Should they be encoded 'at the same time'.. in a way they should share equal contibution to the lkd / recon, no?</span>


Need to get a bit more technical about what exactly do i want to see here in the experiments and the goal
When I go to Jilles what do I want to show him ?

Can my architecture preserve the DAG relationships? --- and where exactly are these enforced?
Right now it seems that the network just encodes both factors in 1dim.. as a combination... it has no concept of 'relatedness via causation'. It just understands relatedness via co-occurence. I need to verify and test this.

- [ ] clarification of semantics and role of $\epsilon$ and their use for creating $z$ with the GNN
- [ ] thinking on what concrete progress can be made wrt Friday meeting - what experiments can be done and what stuff can be implemented and tested
- [ ] clarify again the intervention section and what exactly are they doing with it