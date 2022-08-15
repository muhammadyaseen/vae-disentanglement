Disentanglement / Factorization pressure in GNN based priors / posteriors ?

Relating Graph Neural Networks to Structural Causal Models https://arxiv.org/abs/2109.04173

### GNN idea exploration
- I can use GNNs to implement the latent structure in encoder and decoder. (Why would this be better than current impl? I think this needs to be used in conjunction with other arch. improvements. This alone won't cut it)
- Serial nature of Ladder-like networks where values are assigned sequentially dependent on previous values vs the sync nature of GNNs in which nodes are updated at once. (This seems like a 'surface level' problem.. should be able to get around that)
- DAG has directionality but GNNs don't afaik. I think we can simply implement it using DAG Adj. Mats.
- In GNNs node features are multidim but I want to associate each node with a single distinct concept. How would that be achieved?
- We can have arbitrarily complex message NN but at the end all node features need to be condensed down to same number of dims. Could this lead to a bottleneck ?
- How will the nodes that are on the same "level" be organized into a layer? I'll have to concatenate the features for nodes that belong to the same layer ?
- How do I bridge the gap between node / node features and their use as multi-dim gaussian. I can concat all the nodes and pass it as input to the decoder for creating an image.
- Implementing this in a way s.t. node features have accessed to multi-scale processed X.
- Why do we even need decoder GNN ? 

## Bifurcated features in GNN nodes?
There's a connection to [[Orthogonal Subspaces Idea]]. 

We can bifurcate the GNN features and messages into two parts: 
1. Independent part
2. Dependent part

Strictly speaking, the independent part doesn't really need GNN network and it simply an MLP network since we are not using the adjacency matrix / graph structure when computing it. This part captures local node specific information only.

The dependent part uses adj. mat. and uses other nodes to update itself. This is where the correlations between latents can be captured / encoded. When the DAG shows no connection / parent-child relation for a node, this part will be $0$.

The vision is to be able to manipulate the latents both in a dependent and independent way.

When traversing the Independent part no other factor should change.

When traversing the Dependent part, the concept itself and its children can change in the output image.

Need to have an info theorectic / optimization / framework driven need for it

## What do I mean by localizing the dependency / correlation to some dimensions

How does it plays out in an example scenario ?

Smiling an Eyes_Open are two concepts.

The dependent dims are used to "tune" the details based on parent variable's value.

Independent dims hold more global information e.g about position and color etc.

This analogy doesn't work in DSprites data because in that dataset the latent concept (e.g. posX) only have 1 degree of freedom and is fully specified by one parameter. This is not the case in CelebA where we have more complex latents e.g. "Smiling"
