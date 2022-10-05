noise / jitter in gen process

penalize if posterior node is informative of other concepts?

supervise the prior with actual gt latents?

should the initial node feats produced from cnn be regularized to be independent or each other?

run same kld weighting experiment on switch data? doing

give more capacity to prior?

traverse two nodes at once using fixed val + random traverse method

Implementation with fixed gt based prior.


doing covariance over batch will create gradient relationships b/w different $X_i$'s .. dont know how that will play out in grad opt

i want the nodes to store specific information.
for this i am (1) supving them with actual values in KL and (2) SupReg with actual values.

right now the agg function is just sum.
i need layers to propagate node msgs.. i need to mult with $A$ at least.

Penalize global structure of $q(u)$ using information from $p(z)$.
- already tried penalizing Covariance per batch
- can also try penalizing jacobian

$u = f_\theta(z)$

$q(u) = p(z) \times \vert \text{det} J_{ u \rightarrow z} \vert$

structure of Jacobian should reflect structure of $A$ mat and covariance relations in the ground truth. If it doesn't then it is storing information across the nodes or not storing it correctly...

But how can we use this? we have labels .. how to use this ?