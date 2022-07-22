All dimensions of $\Sigma$ associated with variational distributions go to $0$ around 15th epoch. This basically means that the images are being encoded onto a very small region of latent space. The encoded dist is very compact. We can imagine it as a "bed of nails" kind of configuration where the tip of nail corresponds to the mean represenation associated with the example $\textbf{x}_i$. 

This does mean that each $\textbf{x}_i$ gets a unique unambiguous representation $\mu_i$ which results in ease of decoding as it can't be confused with the representation associated with some other example and results in lower recon error. What we hope is that examples

Dimenstions of $\Sigma$ going to zero is also consistent with the observation in Hidden Talents papers which states that in VAE variance associated with useful dims goes to $0$ and useless dims goes to $1$. 

The weird thing is that I haven't seen ANY dim approaching $1$ std in the posterior or prior. They all go to zero.

Imp thing to note is that in Hidden Talents paper they're working with a fixed Indepented Normal prior whereas in this case we have a learnable non-independent prior. So to what extents the results might hold needs to be thought about.

Prior and posterior variances both go to zero and their trends track each other. It seems liked they're essentially coupled and become "the same" as far as variances are concerned and the only difference / KL-div can only be due to different means.

![[std_behaviour.png]]
