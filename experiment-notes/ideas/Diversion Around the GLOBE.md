Big Picture: Leveraging Causal Discovery for Deep Hierarchical Latent Variable Models

#### Questions about GLOBE
- How many vars can GLOBE run on / run-time ?
- give that we can extend GLOBE to binary vars required for CelebA dataset.. what can we accomplish with it ?
- GLOBE algo also gives the SCM i.e. functions / mechanisms relating the (latent) variables, how can this be used in our VAE models ? Could we somehow use it to regularize the learning?
- I want to convince myself that this GLOBE diversion is worth the time investment.
- Decision Tree / PolyMARS for binary function learning.
- Assuming PolyMARS works... what now ? and should I really be working on that extension?


#### Questions about modeling decisions / outcomes in VAE
- Where do GNNs fit into all of this? ([[GNN Idea Exploration]])
- we might be able to learn much larger graphs and hence we can show a lot more variation / simultanous changes e.g. they learn a maximum of 4 concepts
- they show one pair of cause,effect changes at a time for ex smile and mouth open change i.e just 2 vars changing. we can change multiple at the same time... if we intervene of several nodes at once.
- MoG modelling for individual units ?

#### Questions about CausalVAE and Indept Causal Discovery methods

[[CausalVAE]] already includes a "pre-train stage" to discover the matrix with a continuous objective and DAG constraint (like NOTEARS, [[DAG-GNN]]). How would using any other method be better ? Couldn't we simply run CausalVAE on data with larger DAGs? We will probably have to do it for comparisons.

<u>What do the 3 DAG constraint approaches have in common ? </u>
NOTEARS, DAG-GNN, CausalVAE both use a continuous DAG constraint in the objective. DAG-GNN presents a different more computationally feasible formulation of constraint introduced by NOTEARS. This reformulation is then adopted by CausalVAE.

<u>Could this constraint / optimization be used to discover large DAGs ?</u>

#### Questions about connection between latent labels and latent concepts

Assume we learn the structure and even mechanisms relating the <u>latent labels</u> via some algo e.g. GLOBE. Now we need to explore the <u>connection between latent label and latent feature</u>.

Let latent label(s) we (shape=1, X=20, Y=10).

Technically speaking, for perfect recon we would like the model to encode this example into a unique and well-separated point in the latent space s.t. the reconstruction is then unambigous and perfect. This can be called the "bed of nails" representation.

However, such a latent space is not useful because there's no structure to it other than unique codes. There's no geometric relationship between the codes. For ex, we'd want examples that look similar to be encoded closer together. Another desirable property is that the latent space is structured e.g. moving along the latent manifold leads to predictible changes in the output shape. This won't be possible in the "bed of nails" representation. We need smooth dynamics in latent space.

Latent label is not the same thing as latent concept that it represents. We want the latent space to encode the conceptual variation and not just the latent label. For example we might have a label `smile=True, gender=Male`. But obviously, there's a lot of variation within this class as well. There are many different kinds of smiles and many different kind of male features. We need a way to capture this uncertainly or variation in the concept distribution. This is what we want to model in the latent distribution dynamics. Later on once we have these distribtions, we can annotate individual samples from these distributions. E.g. imagine a distribution of lip features as a 2-MoG where sample from one component can be labelled `smile=True` and from other `smile=False` with some gray area in the middle where mixtures have same likelihood. Analoguous things can happen in the `gender` label.

Latent labels are certainly a function of latent concepts uncovered / encoded by VAEs (e.g. $u_i^{(k)} = \mathbb{E}[f^{(k)}(z_i)|X]$  ) but are not the same thing. Hence we need to think that to what extent can we use the learned mechanism (via GLOBE) behind latent labels to help with latent concept distributions.




