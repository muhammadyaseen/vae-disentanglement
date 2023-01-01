Title: VAEs with structured latent concepts

The last paragraph of the Introduction should inform the reader  about the flow of the thesis and its structure e.g. which chapter contains what.

Every chapter begins with a short paragraph that describes the flow and content of the chapter (somewhat like an executive summary). Reader should have an overview of the chapter and know what to expect after reading this first paragraph.

## Acknowledgements

I have made this list so that I don't forget any 'required' acknowledgements.

1. MPI-Informatik / IMPRS (funding/scholarship + workspace)
2. Koshish Foundation (funding/scholarship)
3. CISPA (workspace + infrastructure)
4. SaarlandStipendium (funding/scholarship)
5. OpenSource community for a lot of code that i built on top of
6. HAI CORE (infrastructure for experiments)
7. Other usual stuff (Advisor / EDA members / others)

## Abstract

In this thesis we address the problem of learning the distribution of latent variables from observed images. Our focus is to preserve the structural constraints present in the original latent distribution. This is beneficial in cases where the structure represents a physical or real-world dependency. Entanglement of such structure in generative modeling would lead to non-representative samples from the learned generative distribution.

To this end, we first present a formal model of generative images from latent variables using Structured Causal Model (SCM) theory. Then we introduce Variational Auto-Encoder to approximate the distributions involved in this generative process. Furthermore, to addresses the capacity and gradient flow limitations of existing Graph Neural Network and Hierarchical modeling approaches e.g. LadderVAE, we propose a novel approach to modeling the latent network in VAEs that does not suffer from such limitation. Next, we instantiate our VAE model with appropriate regularization terms.

We evaluate the performance of our algorithm on two standard benchmark data sets and introduce a third new data set with more complex generative mechanisms. Our experiments show that the algorithm successfully learns the latent distributions that preserve the structure while maintaining excellent reconstruction performance. We demonstrate this by performing intervention experiments which show that changing the parent node affects the child node and therefor preserves the causal relationships present in the original data generating process.

## Introduction

## What to inclue in Prelims / Background?

 This chapter will define the notation to be used throughout the document and will provide definitions of relevant concepts required for understanding the thesis (pre-supposing _some_ prior knowledge). In a way, this chapter is an attempt to make the thesis self-contained.

### A running list of required notation

| Symbol | Comment / Concept |
|--------|------------|
| $\mathcal{U}$ | Set of all labels |
| $\textbf{u}_i$ | Vector of Labels associated with $i$-th datapoint |
| $u_i^j$ | Individual $j$-th label or $j$-th dimension in the vector $\textbf{u}_i$ |
| $u_i^{-j}$ | All dimensions of $\textbf{u}_i$ except $j$-th dimension in the vector $\textbf{u}_i$ |
| $\mathcal{D}$ | Denotes the dataset with labels |
| $\textbf{x}_i$ | $i$-th image in the dataset|
| $\textbf{z}_i$ | latent code associated with $i$-th image in the dataset|
| $z_i^j$ | Individual $j$-th label or $j$-th dimension in the vector $\textbf{z}_i$ |
| $z_i^{-j}$ | All dimensions of $\textbf{z}_i$ except $j$-th dimension in the vector $\textbf{z}_i$ |
| $A$ | adjacency matrix |

1. A short paragraph introducing the flow and content of the chapter.
2. Notation. (see table following this section for a tentative list)
3. What is a Representation and Representation Learning (RL) ?
4. Some desirable properties of representations e.g disentanglement, decomposition, modularity etc + Reference to Causal RL (as they relate to concepts that we plan to demostrate later through our method)
5. Theoretical benefits of those properties in terms of sample coplexity, interpretability etc.
6. Intro and motivation of Variational Inference in general (structured VI?)
7. Introduction and motivation of (NN based) VAE
8. VAE as an RL algorithm / What are DLVMs?
9. VAE / ELBO objective in its simplest independent prior form + Motivation / Need for Lower Bound
10. Role of KL divergence b/w $q(z \vert x)$ and $p(z)$ and Reconstruction $\log p(x \vert z)$ parts in the objective
11. Role of Prior $p(z)$ in VAEs for encouraging various properties (this is crucial, as we exploit this in our method)
12. Hierarchical VAEs ?
13. ELBO in the form used by us i.e. non-independent priors
14. Covariance and Jacobian determinant (I want to mention these concepts here because later we penalize covar / det in the objective. Should I include the motivation for their penalty as well here or just the definition? I think the motivation for penalty should go in the Theory part. Here I can just define the general concept?)
15. Role of Disentangled prior $p(z)$ and Disentangled Variational learned Posterior $q(z)$  in encouraging decomposition of nodes in VAE.
16. Graph Neural Network (used to represent prior / posterior in 1 variant of our alg.)
17. DAG learning from labels ? Causal graph identification from labels? Motivation to relegate this to external specialized methods.
18. Structural Causal Models?

I'm not including any definition / introduction to Neural Nets or Gradient Descent as I assume that the reader is familiar with them already.

I think that points 10, 11, 13 and 15 could go into Theory chapter. These points don't cover 'background' but rather play an important central role in the algorithm and are somwhat specific to our proposed model.

Should I include why we don't use other density learning algs e.g. Norm Flows, Diffusion, AEs, GANs? I think this can go in discussion where I can connect these algos to different parts of VAEs and mention to what extent they could be used and what limitations they have wrt representation learning in our setting ?


## Theory

1. A short paragraph introducing the flow and content of the chapter.
2. This will include formalization of the problem and assumptions and setting etc but won't include solution part. It should be a "what do we want to do" and not "how we do it", that comes in the next (Algorithm) chapter.
3. Should I include Image SCM in this - this is the formalism we ended up using, didn't we? The current architecture (and other approaches e.g. ladder, gnn) can be seen as an instantiation of this idea.
4. Should I include LadderVAE / Network structure / GNN based approach? 

Independence of mechanisms -- it plays a role in our method

Our arch / latent network is like a NN with a very high drop out probability. 

I should also desribe the data generative / latent model somewhere. Where would it go? In theory or prelim or algo?

Connection to DIP-VAE?

### Theory: The Problem

- What do we want to preserve? We want to preserve the semantically meaningful dependence relationships in the data. We want to learn latent variational distribution which respects the known (in)dependence structure in the data.
- It can also be said that we want to preserve the independence mechanisms? SCM formalism gives us a way to talk about these mechanisms a bit more concretely.

Given: D, U, `adj_algo`, hyper params
Output: Representation, Learned Variational posterior with some properties
How:

## Algorithm

In this chapter we describe the proposed architecture and objective function terms that solve the problem we set up in the previous chapter. We essentially have three instantiations, one of which seems to work much better than the others.

1. A short paragraph introducing the flow and content of the chapter.
2. Posit the form of prior
3. Posit the form of posterior
4. Schematic of which parts of architectures correspond to which terms in equations
5. Encoder instantiation (arch details in appendix)
6. Using labels set $\mathcal{U}$ to get adj matrix $A$ or use given $A$.
7. Construct latent layer architecture $f_A(.)$ given matrix
8. Setup prior and posterior given matrix and labels
9. Decoder instantiation (arch details in appendix)
10. Setup ELBO given above 
11. Setup (covar, det) penalization terms given $A$ matrix and labels $\mathcal{U}$
12. Combine above to get the final objective

Feels like some parts should go in Theory.

In a way, Ladder / GNN / NS based approaches are all different instantiations of our idea.

## Evaluation

1. Mechanism preservation experiments on Pendulum
2. Mechanism preservation experiments on Pendulum-Switch
3. Mechanism preservation experiments on Water-Flow
4. Mechanism preservation experiments on CelebA. I'm kinda curious about what we can get out of CelebA
5. Role of Capacity shown on any one of the datasets
6. Compare with Causal VAE / BetaVAE
7. Curious: 2d sprites on SLC-VAE and $\beta$-VAE
8. Latent space evolution and plots
9. Ablation: Supervised Regularizer
10. Ablation: Covariance Reg
11. Hypothesis: The two islands that we sometimes see in latent space correspond to indpendent combinations of the two latents. e.g one islands encodes (1,0) and the other (0,1) or in words (red switch, off light) and (green switch, on light).

## Discussion
- connect my early experiments on three shapes dataset.
- continuous and smooth changes in latent space may not correspond to continuous and smooth changes in the data space e.g. light chaning from green to red.
- In many current synthetic datasets, backgrounds are mostly static (2dsprite, waterflow, pendulum). This way model can get a huge reduction is error just my correctly memorizing the background. This plays no role in actually learning about the latent factors.

- Connections to:
	1. Broken ELBO, 
	2. Yet Another Way to Carve ELBO
	3. Don't Blame ELBO
	4. PCA and VAE
	5. Channel Capacity
	6. Hidden Talents of VAE etc?

- Connection to classical "Parts Model" - This was one of the motivation for ImageSCM idea
- Conditional HSIC regularization for independence (A better alternative to KL-div?) + connection to Krik's paper
- Semantic Segmentation weighted loss for recon (Current recon loss doesn't respect 'closeness' in image space which might contribute to not respecting closeness in latent space)
- Trade-off between Recon and KL-div and commensurability of these two parts
- (Related to above) Connection to MDL two-part codes
- Relation to I-map or D-map of the network structure and the (in)dependencies we want to have.

Variation in any dimension of $x$ could potentially be because of a latent variable. Pic of bycicles that has same colors technically then lacks the red latent var. So in a sense what variations we can capture definitely depends on the dataset we have.


**Improvements and Extension**: 
1. Incorporating or adapting any of the tricks useful for Training large VAEs e.g. those mentioned in NVAE. Hidden Talents of VAEs
2. Datasets are kinda shit. What other interesting datasets could there be ? Non-image data?
3. Right now we're penalizing Cov mat which only takes care of dependence relationships. We can also try penalizing Information Matrix form as well to take care of conditional independencies. (Add ref to chap and thm)
4. Extending to include scenarios where we have some protected and some uprodected attributes i.e we want to preserve some relationships and disentangle some other set of relationships because they might represent bias etc.

## Other Required Stuff

- **Any work** performed on the HAICORE@FZJ partition should be **appropriately cited** as "This work was supported by the Helmholtz Association's Initiative and Networking Fund on the HAICORE@FZJ partition."
- TODO: Make a list of such required attributions as I go along. For ex, see if PyTorch, TensorFlow, Matplotlib, `disentanglement_lib` or any other software / hardware infrastructure requires some specific attribution or citation.

