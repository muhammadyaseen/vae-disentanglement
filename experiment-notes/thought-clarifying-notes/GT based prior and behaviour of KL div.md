Normally when the prior is $\mathcal{N}(0,I)$ i.e. completely uninformed the sample-wise KL divergence $KL(q_i \vert \vert p)$ represents channel capacity i.e. how much info the network is distilling from $X_i$ into $z_i$. If the KL div is zero then $z_i$ is un-informative for (and indepedent of) $X_i$ .

But what if we have a point-wise informed prior?

For a pair $(X_i, z_i)$ we have $p(z_i)$ that encodes noisy 'true latents'.
In that case if the $KL = 0$ even then the latent $z_i$ is informative because it uniquely determines the $X_i$. The network can just match the prior params to get zero KL. 

Doe this tie in with the idea that best prior is "marginalized posterior" $q(z)$?

If $KL > 0$, then that could mean that the encoder is distilling information in a non-gaussian way... this could even imply a kind of inefficiency.


if the prior and posterior both have learnable params then the situation becomes tricky optimization-wise.

What I'm observing right now when using learnable GT based prior is that the network collapses whole dataset onto a single point in all the nodes until about ~40 epochs but then "the floodgates open" and it starts to allocate unique mean points to different $X_i$. Why is this initial collapse happening? and what causes the network to recover from it after ~40 epochs?

Could it be that since i have fixed the prior $\sigma$'s the only way for opt to reduce KL is to collapse together?
In current network for first 40 epocsh posterior std stays above 1 which isn't good  because it means very noisy samples.. and overlapping in latent space.

Why does this not happen with fixed prior? (i can't sat that it doesn't happen because i haven't really experimented to see it).. There the nework can collapse stds and spead means to reduce KL.



