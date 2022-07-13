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