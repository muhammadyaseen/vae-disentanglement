#### Posterior variance
All dimensions of $\Sigma$ associated with variational / posterior distributions go to $0$ around 15th epoch. This basically means that the images are being encoded onto a very small region of latent space. The encoded dist is very compact. We can imagine it as a "bed of nails" kind of configuration where the tip of nail corresponds to the mean represenation associated with the example $\textbf{x}_i$. 

This does mean that each $\textbf{x}_i$ gets a unique unambiguous representation $\mu_i$ which results in ease of decoding as it can't be confused with the representation associated with some other example and results in lower recon error. What we hope is that examples

Dimenstions of $\Sigma$ going to zero is also consistent with the observation in Hidden Talents papers which states that in VAE variance associated with useful dims goes to $0$ and useless dims goes to $1$. Here both dims go to $0$ so it'd mean that both feature dims associated with a node are being used.

Imp thing to note is that in Hidden Talents paper they're working with a fixed Indepented Normal prior whereas in this case we have a learnable non-independent prior. So to what extents the results might hold needs to be thought about.

#### Prior variance
Accross all 5 nodes the following trend:
1st dimension of prior variance goes to zero, although it does so slower than the posterior variance. 
The 2nd dimension is non zero and is close to $1.2$ mark is all nodes. 

It is strange that in all the nodes only the 2nd dimension is non-zero. If the network were to randomly choose via optimization which dim to use we'd expect to see at least one case where 1st dim was used instead of 2nd.

We can say that the KL-div contributed to the loss is caused by this non-zero dim of prior variance.

![[std_behaviour.png]]

#### Prior and Posterior $\mu$

The 1st dim of both prior and posterior is very close to zero. For posterior it is very close to the zero and for prior it is some small -ve value e.g $-0.15$

The 2nd dim is interesting:
1. In the prior it has a spread from $-0.4$ to $+0.4$
2. In the prior it has a spread from $-4$ to $+4$
So in both cases it is not dead like the 1st dim but it is again surprising in all the nodes 2nd dim somehow survived. This seems too systematic to be just chance.

We can again say that a larger contribution to KL-div comes from 2nd dim.

![[mu_behaviour.png]]

Why is the prior variance 2nd dim not being pushed to zero..?
Why is the post variance BOTH dims being pushed to zero..?

Why the scale difference b/w prior mu dim 1 and post mu dim 1 ?

