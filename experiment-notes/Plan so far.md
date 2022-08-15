- In many ways [[CausalVAE]] is not sound.
- The latent SCM model / equations it uses are limited. We can extend these to more permissive and expressive latent interactions.
- They model only the concepts for which they do have labels. There's no mention of what happens to the latent concepts that are present in the data but aren't modelled. We can clarify this part of generative story and modelling.
- They learn DAG while learning the concepts and use labels for supervision. These two tasks can be decoupled i.e. we can learn the DAG using any available Causal Discovery method and then just use the structure when training
- The above point also strengthens the claims we can make about the structure we are enforcing. Afaik DAG-GNN and NOTEARS method of learning the DAG are limited in the assumptions
- We can use GNN to more naturally model the latent dependencies and simultaneously allow more complex interactions.
- Since we use labels to learn the structure and we want to make a distinction b/w labels and conceps we need to clarify the difference between two an justify the use of labels for learning concep DAG.
- They demonstrate their interventions on 2 concepts at a time for a total of 4 concepts. We can potentially learn larger DAGs and intervene on multiple nodes / sub-graphs.
- The Idea of Image SCM and interpretation of different variables used in modeling

