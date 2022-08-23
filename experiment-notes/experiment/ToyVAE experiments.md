### ToyVAE

2 Types of Data

### Strong Correlation
- Without Loss wieghting was getting terrible recons. So I moved to no-correlation. Got the following after loss weighting
- Stable KLD and recon
![[recon_kld_stable_weightedeblo_corr.png]]
- Dimensions still very close to zero
![[e100_corr-toy-recons-scalars.png]]

Reasonable recons
![[e100_corr-toy-recons.png]]


### No correlation

- Used $5$ as recons weight and $0.1$ as KLD Loss weight. No KLD Scheduling
- It took 100 epochs to get to very good recons. Output after 50 left, 100 right 
![[e50_e100_uncorr-toy-recons.png]]

- Because of the weak KLD loss, there's no discernable structure in the latent distributions![[e50_e100_uncorr-toy-hists.png]]
### Scale and Architectural Considerations related to HVAEs
Spatial latents vs. Concepual latents

- I'm running into problems associated with training large scale (hierarchical) VAEs. 
- It seems that HVAEs e.g. Ladder VAE, BIVA etc have used rather wide latent layers that definitely can't correspond to concrete conceptual latents. They instead use the concept of latent groups associated with spatial dimension i.e. higher / wide latents in earlier layers where the input size is large (this is the result of `Conv2D`) and fewer latents in later layers.

Niether CommAss nor LadderVAEs trained on CelebA. (Which networks did train on CelebA ?)

- One of the problems with current implementations is that they don't respect the typical CNN hierarchical feature extraction assumption. Because of DAG structure the BottomUp Networks can be of arbitrary width. In normal CNNs the network width gradually decreases, but in BUNets it doesn't follow any predictable pattern. Subsequent layer can have lower or higher width depending on "concepts" or "nodes" in corresponding DAG layer. I need to separate hierarchical image features and conceptual latents. It feels that the network is having a hard time extracting and remembering anything meaningful. BU layer is not able to extract anything meaningful. 

- If we only have 1 root node it introduces severe bottleneck. Need a principled way to deal with it.

- In DSprites for example, I'm projecting down the whole image $X$ into $2$ units in the first BottomUp layer (1 unit for $\mu$ and the other unit for $\Sigma$, then the next layer projects up to $4$ dimensions. However, the information has already been lost. The DAG structure doesn't correspond to how NN's process information. A careful rethinking of architecture is needed that respects DAG constraints while also being mindful of NN information flow and how it processes it.

- Getting a strong urge to build a small toynet with known correlated generative process and then apply a simple VAE to it. Least then I will be able to make clear sense of stuff and have more control over "latent-concep-to-unit-mapping". DSprites, while simple is still far to complicated of a dataset to try a whole new architecture on. This has an additional advantage that I won't run into VAE scaling issue.