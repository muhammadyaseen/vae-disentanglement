- In many ways [[CausalVAE]] is not sound.
- The latent SCM model / equations it uses are limited. We can extend these to more permissive and expressive latent interactions.
- They model only the concepts for which they do have labels. There's no mention of what happens to the latent concepts that are present in the data but aren't modelled. We can clarify this part of generative story and modelling.
- They learn DAG while learning the concepts and use labels for supervision. These two tasks can be decoupled i.e. we can learn the DAG using any available Causal Discovery method ([[Diversion Around the GLOBE]]) and then just use the structure when training
- The above point also strengthens the claims we can make about the structure we are enforcing. Afaik [[DAG-GNN]] and NOTEARS method of learning the DAG are limited in the assumptions
- We can use GNNs to more naturally model the latent dependencies and simultaneously allow more complex interactions ([[GNN Idea Exploration]])
- Since we use labels to learn the structure and we want to make a distinction b/w labels and conceps we need to clarify the difference between two an justify the use of labels for learning concep DAG.
- They demonstrate their interventions on 2 concepts at a time for a total of 4 concepts. We can potentially learn larger DAGs and intervene on multiple nodes / sub-graphs.
- The Idea of Image SCM and interpretation of different variables used in modeling [[Image SCM Idea Formalisation]]

### Questions on above points
Figure out a way to run CelebA with all / limited DAG
What is the right protocol to train when using a learnable prior e.g. NVAE for training tricks
Taming VAEs https://arxiv.org/pdf/1810.00597.pdf 
Hidden Talents of VAEs https://arxiv.org/pdf/1706.05148.pdf
Do I have any prev expr runs with gnn based prior on celeba?
Color images should use MSE Loss ?

