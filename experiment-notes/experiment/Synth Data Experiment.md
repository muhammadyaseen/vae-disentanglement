![[Pasted image 20220816101736.png]]

Interesting to see that the reconstructions cover the whole support of the image. It probably means that the code learned isn't very informative. 

In this run the weights were (1,1). KLD loss went to 0 for all the nodes. So there's no information coming from input image to latents. Latent layer is essentially just noise and retains no information.

