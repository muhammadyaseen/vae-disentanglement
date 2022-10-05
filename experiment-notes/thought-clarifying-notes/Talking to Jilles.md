### About thesis

- Reason for diversion: The ideas and goals were a bit vague and there was no story / whole to organize the thought behind project. 
- I found 2 competing works that turned out to be basically one work i.e. DEAR and CausalVAE
- They have very similar setting but I (on VAE end) and Osman (on Causal setting end) think that it can be done better and that our formalism is clearer and sound (per Osman)
- I consulted with Osman to ask his opinion and guidance on the causal setting used in CausalVAE and he had a lot of things to say. Essentially, he also thinks that we can situate our work in a more robust setting and improve the formalism.
- I have run experiments on one of their synthetic dataset and the results look good. The causal relationship is preserved. Now I have to do more systematic experiment and also on other datasets
- Ask for seminar and / or registering? Imp to get a mental timeline. But I also want to start with Isabel that means more work so.. might backfire?

### Plan and comparison

- In many ways [[CausalVAE]] is not sound - their proofs / claims are for linear case but they use non-linearities etc.
- They introduce SCM only at the level of latents. This seems a bit adhoc. Instead we can interpret the whole process as an Image SCM [[Image SCM Idea Formalisation]]
- The latent SCM model / equations it uses for latent are limited. We can extend these to more permissive and expressive latent interactions.
- They model only the concepts for which they do have labels. There's no mention of what happens to the latent concepts that are present in the data but aren't modelled. We can clarify this part of generative story and modelling.
- Related to above point: They implicitly assume causal sufficiency since they use all the labels they have and don't model latents w/o label. We can address this.
- They learn DAG while learning the concepts and use labels for supervision. These two tasks can be decoupled i.e. we can learn the DAG using any available Causal Discovery method ([[Diversion Around the GLOBE]]) and then just use the structure when training
- The above point also strengthens the claims we can make about the causal structure we are enforcing. Afaik [[DAG-GNN]] and NOTEARS method of learning the DAG are limited in the assumptions and function + noise classes they address.
- Once we have the structure we can use GNNs to more naturally model the latent dependencies and simultaneously allow more complex interactions ([[GNN Idea Exploration]]). This would be a generalization of their `CausalLayer` or `SCMLayer` idea.
- Since we use labels to learn the structure and we want to make a distinction b/w labels and conceps we need to clarify the difference between two an justify the use of labels for learning concep DAG.
- They demonstrate their interventions on a max 2 concepts at a time for a total of 4 concepts. We can potentially learn larger DAGs and intervene on multiple nodes / sub-graphs.
- The Idea of Image SCM and interpretation of different variables used in modeling [[Image SCM Idea Formalisation]]
