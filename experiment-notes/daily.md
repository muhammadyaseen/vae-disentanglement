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

As SCM can be written as a composition of functions


