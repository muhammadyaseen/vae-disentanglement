**Any work** performed on the HAICORE@FZJ partition should be **appropriately cited**: 
"This work was supported by the Helmholtz Association's Initiative and Networking Fund on the HAICORE@FZJ partition."

Every chapter begins with a paragraph(s) describing the summary / what to expect in the chapter. One should have an overview of chapter by reading this. It describes the flow and content of the chapter.

## Acks
1. Koshish Foundation + MPI + CISPA + SaarlandStipendium + OpenSource + HAI CORE + usual

## What to inclue in Prelims?
1. Notation and definition of imp concepts
2. What is a Representation and representation learning?
3. Some desirable properties of representations e.g disent, decompose, modularity etc?
4. Benefits of those properties in terms of sample coplexity, interpretability etc.
5. VAE architecture and its motivation as a repr learning alg + connection with AEs?
6. VAE / ELBO objective in its simple independent prior form?
7. Role of KL divergence and Recon parts
8. Role of Prior in encouraging properties
9. ELBO in the form used by us
10. Covariance and Jacobian determinant (should I include the motivation for their penalty as well?)
11. Role of disent prior and disent variational learned posterior
12. Should I incl why we don't use other density learning algs e.g. Norm Flows, Diffusion, AEs, GANs?

## Theory
1. This will inclue formalization of the problem and assumptions and setting etc but won't include solution part. It should be a "what do we want to do" and not "how we do it", that comes in the next (Algorithm) chapter.
2. Should I include image SCM in this - this is the formalism we ended up using, didn't we?
3. Should I include LadderVAE inspired approach?
4. Should I include Network structure based approach?
5. Should I include GNN based approach? 

## Algorithm

1. Using labels to get adj matrix / use given adj matrix
2. Construct architecture given matrix
3. Setup prior given matrix
4. Setup ELBO given matrix
5. Setup penalization terms given matrix and labels

## Discussion

Connections to Broken ELBO, Yet Another Way to Carve ELBO, Don't Blame ELBO, PCA and VAE, Channel Capacity, Hidden Talents of VAE etc?


