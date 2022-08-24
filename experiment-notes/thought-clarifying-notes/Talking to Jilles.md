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

<span class="remark">At this point I feel that my experimental work is severely lacking. I should have couple of experimental runs on all the relevant datasets. Need to integrate MPI-3D, Chairs etc dataset and run experiments on them...? But they're independent.. so either I introduce correlation or use them for something else</span>

Google's `disentanglement_lib` has loaders built in for these datasets. I should use them to save time.

### About working with Isabel
- Why do I want to spend some time in that group? EDA groups works on foundations and it seems that ML group works on applying those foundations at ML-scale models... My interest it feels that is somewhere in the middle. Probably also reflects in my thesis work so far. So I want to explore their work a but more closely.
- Find 3-4 of her group's papers that I like and read thru them at least in broad strokes (VACA, Algorithmic Recourse)
- When can I start and how do I get introduced to her
- PhD contract info that he might have ?


### Points / Review of last meeting

He doesn't seem to worried about even having a seminar. I get the feeling that I'll not have a proper seminar presentation and it will be merged with defense or something like that.

He mentioned that in his mind he has already "registered" my thesis "some time in the past" and that any actual official registeration will have to be back dated by a couple of months. I have the feeling that mentally he has the timeline of 2-3 months for finishing everything up. This also align with what I personally feel as well.  I should be done in next 3 months give or take 1-2 weeks.

Above two points mean that I need to do a lot of experiments quickly and overall iterate quickly on the ideas to get something decent out.

On this note, he suggested that first we can find out a case where CausalVAE doesn't pick up / or doesn't do well and our architecture does. (I need to think a bit about this and see how different disent papers etc have compared with each other). Then do a kind of Ablation experiment for our architectures.

By next week Friday, I need to have done more experiments of Pendulum and Flow dataset. I'll think up how to handle Celeb dataset and how to compare the two approaches.

Another interpretation of epsilon -> z mapping: epsilon's represent independent attributes whereas z's represent attributes that have a DAG-y structure