- [ ] Read up on Isabel's group's work
- [ ] Approaching Isabel
- [x] Approacing Adrian for more in-depth info on current research and future directions he sees the group moving to in terms of topics
- [x] Writing up a formal version of current work
- [ ] Can I think more about my MDL + VAE idea as a possible project pitch to Isabel? [[Minimal Cost Latent Structure Search]]
- [ ] Checkout disent lib's datasets
	- Checked: can easily integrate MPI-toy-3d, Cars, SmallNORB datasets
	- But all these datasets are with indept latents so they don't really help in the correlated case
	- we can use them to test the loss weighting thing
	- probably need to find more managable correlated datasets. I can show results on Flow, Pendulum, and CelebA etc. but for later i'll need more
- [ ] running exp on celeba
- [ ] thinking about bits of celeba and related datasets
- [ ] related cross ent and mse loss for images
- [ ] If I implement the latent tracking thing then I can also see the effects of $\beta$ , Recon weight, KLD weight on the behaviour of latent encoding.. Basically keep all the things (batch size, LR, arch, latent dim etc) same and run several experiment with varying settings of these params and save the behaviour of latent space... Then I can visualize and compare
- [ ] Save the recon pictures are every epoch


Goal: After coming from lunch I should be able to see the result of the dynamics plots