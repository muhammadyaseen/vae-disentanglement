Take all samples where only one factor has been changed (Conditional Sampling), pass those samples through the network and record all latents activations. Then compute mean and variance of each latent dimension. Using Z-scores or something similar you could probably easily make them comparable

- [x] Activation Histogram for each latent dim - Done: Discovered dead dims and solved the issue of strong sq_diff in non-recon relevant dimensions
- [ ] Responsibilities for each latent dim when only 1 factor is different / changed. The problem is that because of correlation we can't guarantee that only 1 factor would be different if we sample the correlated factors
- [x] Does the error only occur when using multiple GPUs ? No it also ocurred in 1 GPU case. Further investigation revealed that all the relevant tensors are becoming `NaN` for some reason. One suggested reason was higher LR. Trying with 1/10th LR now.
- [x] Is Using a single GPU actually faster than 4 GPUs ? It does seem so. 1 GPU takes 5m/epoch whereas `dp` with `gps=4` took nearly 9m/epoch.
- [ ] Find saved best model from disentlib and convert to PT
- [ ] Finding out responsible latent dimension by passing in two X's with maximally different latent in 1 dim with other dims fixed (Jonas' msg : Normalize / Z-score etc)

What should happen to the dimensions that DO show recon difference when we perturb them?






### $\beta$-VAE  with $\beta = 1, \sigma = 0.2$ and 26 epochs
The eperiment follows the same protocols as CommAss paper. It seems that the network wasn't converged at this point. Recons also have a room for improvement.

![[bvae_b1_s02_26_loss.png]]
![[bvae_b1_s02_50_recons.png]]

The main reason to train this network was to do perturbation experiments. Those experiment didn't show a pair of correlated dimensions as we expected.

### $\beta$-VAE  with $\beta = 1, \sigma = 0.2$ and 50 epochs

To do perturbation experiments on a converged network, I trianed the above network under same settings for 50 epochs. This time the network seems to have converged.
1. Recon loss only went down by 0.50 units in last 15 iters (from iter 35 to 50)
2. KLD Increased by 0.15 units in that same iters.

**Perturbation experiment**:
	1. Didn't observed correlated dims. When perturbed, only the perturbed dim reflects change on 2nd pass. No substantial change in any other dim, diagonal structure in 4 columns
	2. Only 4 dims seem to change at all. Chages in other dims are on the order of 1e-3 , 1e-4 or even smaller
	3. Feature importances not very impressive / exclusive. Seem to be distributed thru out dimension except for posX

I also plotted pairwise activations with color coding as in paper
	- For this i'll probably first need to find responsible dims for every gen factor: Implemented GBTs to find responsible dims via feature imps
	- The (test) performance of GBT is terrible and the trends aren't as clear as mentioned in the paper. Responsibilities seem to be distributed accross dimensions

** Latent Traversals**
I plotted Latent traversals of highest varying dimensions for $\beta=1$ and epochs=50 network and observed the following
1. There are two sets of dimensions Strong Sq Diff =  {4,5,6,7,9} and Strong Responsibilities = {0,1,2,3,8}. (See image below)
2. The first set shows strong sq diff when perturbed but doesn't show strong responsibilities. 
3. The second set shows virtually no sq diff when perturbed, but does show relatively strong responsibilities
4. To add to the weirdness... The dims that show no change upon reconstruction / no sq diff are getting mapped to zero during the Latent Traversals. I have tried with several achnor images, they always get mapped to zero or very small values in those dims
5. So essentially, Dims that show sq difference show no reconstruction difference, and dims that show very small sq difference show large recon difference. They are basically disjoint, as show in the following image.

![[sq_diff_vs_feat_imps.png]]

- [ ] Train another network for 50 or more epochs but with higher $\beta$ e.g. $\beta=2$ or more. Why? More disent pressure?