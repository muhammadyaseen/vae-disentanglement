- [ ] make notes from experiment results
- [ ] Implement `--continue-training` 
- [ ] read NVAE paper
	- Residual Gaussian dist parameterization
	- KL balancing coeffs
	- KL warm-up
	- Spectral reg ?
	- How BU and TD nets are treated ?
- [ ] Read VACA paper https://arxiv.org/pdf/2110.14690.pdf

Spatial latents vs. Concepual latents

I'm running into problems associated with training large scale (hierarchical) VAEs. 

It seems that HVAEs use rather wide latent layers that definitely can't correspond to concrete conceptual latents. 

Niether CommAss not LadderVAEs trained on CelebA. ( Which networks did train on CelebA ?)

One of the problems with current implementations is that they don't respect the typical CNN hierarchical feature extraction assumption. Because of DAG structure the BottomUp Networks can be or arbitrary width, and if we only have 1 root node it introduces severe bottleneck. In normal CNNs the network width gradually decrease, but in BUNets it doesn't follow any predictable pattern. Subsequent layer can have lower of higher width depending on concepts in corresponding DAG layer. I need to separate hierarchical image features and conceptual latents. It feels that the network is having a hard time extracting and remembering anything meaningful. BU layer is not able to extract anything meaningful.


