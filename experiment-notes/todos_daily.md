- [x] Check the status of components after mix gaussian init training
	- The start out different (upto 10 epochs) but then follow similar trends. The values are the same upto 1 or 2 decimal places. 
- [ ] Think about and formalize the orth subspace idea
- [x] Train with sup reg on
	- training works
- [ ] Think on how train on celeba. 
	- [ ] How to manage initialization of such a large number of nodes? The MoG approach likely won't scale to larger number of nodes
	- [ ] What kind of DAG structure do we have to train for ? the one we learned via trees?
- [x] Implement the cross ent variant of Sup Reg for Celeba
- [ ] Think on Ind Subspace vs. KL div reg and the relation between the two wrt info theory and lin alg


The std of  emperical variational mu_dists have a much larger variance than that warranted by variational logvar . logvar has large negative values e.g. -6 which would mean we have bascially zero variance. that doesn't make sense. But emp. mu dist has good spread from -3 to +3

Running the network on dsprites_correlated with sup reg on, mix init for prior, and 2d node feat dims for 10epochs


what is the right protocol to train when using a learnable prior

Ladder vae has learnable prior - i kinda didn't really check what was the behaviour there.. were we still getting almost 0 std there?

Taming VAEs https://arxiv.org/pdf/1810.00597.pdf 
Hidden Talents of VAEs https://arxiv.org/pdf/1706.05148.pdf