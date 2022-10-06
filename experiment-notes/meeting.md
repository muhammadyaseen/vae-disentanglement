- 10 models w diff kld values
- Model with covariance penalty on `pendulum`
- 1+ 5 models with with a different way to express latent relations on `pendulum` and `pendulum_switch`
- Idea for penalizing distribution complexity and covariance with Jacobian
- non-technical stuff

### Experiments
- `pendulum_nomr_enc_kldw` - GNN / Pendulum dataset / Fixed Prior
- `pendulum_sw_norm_enc_kldw` - GNN / PendulumSwitch dataset / Fixed Prior
- `pendulum_covar_test` - GNN / Pendulum Dataset / Fixed Prior / Covariance Penalty
- `pendulum_latentnn` - NN Latent Impl / Pendulum / Fixed Prior
- `pendulum_sw_latentnn` - NN Latent Impl / PendulumSwitch / Fixed Prior

I trained 10 models in total by taking five different w_kld = 5,10,15,20,25 over 2 datasets.
- pendulum (no switch, just 2 levels)
- pendulum_switch (3 levels)

Covariance penalty with w_kld = 15
With latentNN implementation w_kld = 15 on `pendulum`
With latentNN implementation w_kld = 2,5,25,40 on `pendulum_switch`

Recon weight was set to 1 for all models. So this we just to see KLD effect.
All networks get good recon value / image, so recon weight doesn't seem to matter.

Where do we go from here? I have compute access on HAI_CORE until end of Oct.

### Jacobian / Covariance Penalty idea

Penalize global structure of $q(u)$ using information from $p(z)$.
- already tried penalizing Covariance per batch
- can also try penalizing jacobian

$u = f_\theta(z)$

$q(u) = p(z) \times \vert \text{det} J_{ u \rightarrow z} \vert$

structure of Jacobian should reflect structure of $A$ mat and covariance relations in the ground truth. If it doesn't then it is storing information across the nodes or not storing it correctly...

But how can we use this? we have labels .. how to use this ?

My method might have DIP-VAE as a special case ?

- [ ] What does it mean to be "jointly Gaussian"?
- https://stats.stackexchange.com/questions/309657/jointly-gaussian
- https://people.eecs.berkeley.edu/~ananth/223Spr07/jointlygaussian.pdf
- [ ] Is my var. posterior jointly Gaussian?
- [ ] If so what form do its params take?
- [ ] Can the covar mat from it be used / inverted to see cond. indeps. ?
- [ ] gt batch covar matching vs Adj mat structure matching
- [ ] add some noise in gen process


