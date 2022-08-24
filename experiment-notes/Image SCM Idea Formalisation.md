### Image SCM: Leveraging Causal Discovery for Deep Hierarchical Latent Variable Models
We view the complete generative process of images as an SCM flow.
### 1. Generative story
- Images $X_i$ are composed of low level features arranged in a hierarchical manner. This is similar to the motivation behind classical Parts Models, Visual Bag of Words etc.
- At the lowest level these features represent primitive shapes e.g. (oriented) lines, color shades, simple textures etc. Because of their simple nature these features can be assumed to be independent of each other. In this formalism, we call them  <u>"primitive features"</u> or "<u>primitive latents</u>" and denote them $\epsilon_i$.  <span class="remark">(could we somehow actually sample from such "primitive" distribution? a paper by JT did something similar where the instantiated the prior from examples within the dataset)</span>
- These primitive features are then combined, arranged, and composed spatially to form intermediate higher-level features ultimately culminating in the observed image $X_i$. Such intermediate features are not explicitly named and are assumed to be captured in the hidden layers / convolutionals kernels of the CNN.
- At some high enough level these features assemble into semantically understandable features i.e. features that align with human understanding of the object depicted in the image (e.g. smile, hair color etc.). We call these the "<u>semantic latents</u>" or "<u>conceptual latents</u>" (interchangeably throughout) and denote them by $z_i$ 

| Variable      | Concept |
| ----------- | ----------- |
| $X$   | Observed image        |
| $z$   | Semantic or conceptual latents        |
| $\epsilon$ | Primitive latents   |

### 2. Modeling choices
In this setting the causal model starts from independent top-level nodes $\epsilon_i$ and follows (unknown) generative mechanisms to get semantic latents $z_i$ and then finally to the image $X_i$. 

![[pgm_partial.png|150x220]]

We can summarize the generative process as follows:

$$\epsilon \sim \text{PrimitiveFeaturesDist(.)}$$
$$z = f_{gen}(\epsilon) + \eta$$$$X = f_{arr}(z) + \xi$$
Where:
- Variables $\eta$ and $\xi$ represent noise with approprite dimensions.
- Function $f_{gen}$ introduces correlations between different semantic latents $z_i$ according some true causal dependency structure $A$. 
- We don't assume that this structure $A$ is directly available to us. We do however have access to labels $\{u_i^{(k)}\}_i$ associated with some of the semantic latents and can use them to learn the structure at least partially. (see Sec. 5 & 6) 
- We choose to model this structure via a GNN for the following reasons:
	- GNN lends itself naturally to hierarchical modeling and we can direcly use leanred $A$ to instantiate our GNN architecture's topology.
	- The way GNN transforms initial node features via intermediate layers into final node features while respecting the depedencies naturally applies to our setting where we transform independent primitive features into semantic $z$'s i.e. $p_{\theta}(z \vert \epsilon, A) = \text{GNN}_A(\epsilon)$ for the prior and $q_{\phi}(z \vert \epsilon, X, A) = \text{GNN}_A(\epsilon \vert X, A)$ 
	- This allows us to model more complex non-linear interactions between latents unlike CausalVAE which only uses simple additive linear interactions limited to one-layer. <span class="remark">(though in linear it doesn't really make sense to talk about layers)</span>

With the introduction of noise and labels the model looks like this:
![[images/pgm_full.png|200x200]]
<span class="remark">(this looks a lot like LadderVAE family of DLVMs. Should take a close look as it might help inform other architectural choices)</span>

The dependency between concepts introduced by $f_{gen}$ is modelled as:
$$p(z)  = \Pi_i p(z_i \vert \text{pa}_i)$$
### 3. Mapping model parts to VAE architecture
We can say that encoder captures $\epsilon$ in its final hidden layer, then the GNN uses these low-level features to produce $z$ as final node features. These conceptuals latents are related via a DAG structure denoted by $A$. Given all this, the Decoder then finally produces an image $X$.
- $\epsilon \approx \text{Encoder}(X)$
- $z \approx \text{GNN}_A(\epsilon)$
- $X \approx \text{Decoder}(z)$

To emphasise the inter-relations between $z$'s we can also write:
$$z_i = \text{GNN}_A(\text{pa}_i, \eta_i)$$
where $\eta_i$ is the local noise in the semantic latents.

![[vaearch.png|500x300]]
### 4. Assumptions on the causal latent structure $A$
- We bifurcate our semantic latents into two groups $z_l$ and $z_{ul}$. For $z_{l}$ we know the latent labels and hence can find the latent structure (see Sec 5 below). For $z_{ul}$ we don't have any labels or structure information. 
- This is different from CausalVAE where they assume causal suffeciency. In our model we assume only partial knowledge of the causal structure gained via partial labels. Therefore instead of full causal DAG, we work with learned semi-Markovian Graphs. <span class="remark">(confirm with Osman)</span>
- Using GNNs we can model highly non-linear interactions between our semantic latents. In contrast, CausalVAE only allows linear interactions.
- We also explicitly model latents for which we don't have any label or structure information. Such latents aren't addressed in CausalVAE. <span class="remark">(TODO: where and how?)</span>

### 5. Learning the structure $A$
- Causal VAE learns the structure in a "pre-train" step using latent labels. <span class="remark">(Do they turn the gradients off later?)</span>.
- After formally justifying the use of latent labels to learn the structure of latent concepts we can completely decouple learning $A$ from the training of VAE. This allows us to use any available Causal Discovery method to estimate the structure e.g. GLOBE. 
- Currently, they use DAG-GNN / NOTEARS based differentiable / continuous objective to learn the structure. But these approaches are limited to very simple function classes and noise settings. <span class="remark">(confirm with Osman) (do they assume causal sufficiency?)</span>

### 6. Using Latent Labels to learn structure $A$ of Latent Concepts
We want to make an explicit distinction between latent labels and latent concepts.

Latent labels are certainly a function of latent concepts uncovered / encoded by VAEs but are not the same thing. (see Sec 7 below). 

<span class="remark">(Given this, we cannot use the learned mechanisms / functions (e.g. via GLOBE) but just the structure, right?)</span>

We assume the following labelling process for our semantic / conceptual latents: <span class="remark">(Might need a non-additive model for binary labels?)</span>
$$u_i^{(k)} = f_{annotate}^{(k)}(z_i) + n_i$$
The function $f_{annotate}^{(k)}(.)$ can for example represent a human labeller that looks at the image and assigns labels to the relevant concept reflected in the image and $n_i$ is the local labelling noise.

![[images/label_and_concept.png|300x300]]

Then any given image and latent concept pair $(X_i, z_j)$ the observed label is the expected value of this function:
$$u_i^{(k)} = \mathbb{E}[f^{(k)}(z_j)|X_i]$$

### 7. Latent label vs. Latent concept
- Latent label is not the same thing as latent concept that it represents. We want the latent space to encode the conceptual variation and not just the latent label. For example we might have a label `is_smiling = +1 or -1, hair_length = +1 or -1`. 
- But obviously, there's a lot of variation within this class as well. There are many different kinds of smiles and many different levels of hair length. We need a way to capture this uncertainly or variation in the latent concept distribution. 
- Later on once we have these distribtions, we can annotate or label individual samples from these distributions. E.g. imagine a distribution of "smiles" as a 2-MoG where sample from one component can be labelled `is_smiling=+1` and from other `is_smiling=-1` with some gray area in the middle where mixtures have same likelihood. Analoguous things can happen in the `hair_length` label.

### 8. Loss terms weighting
Through my own experiments and discussion in literature we know that we need to weight and sometimes even anneal the different terms of loss to get good results. I think there should be a more prinicpled way of doing this. Hence I have been exploring the idea of [[Finding the right weightage for Loss func terms]] in terms of intrinsic Latent and Observed entropy.

