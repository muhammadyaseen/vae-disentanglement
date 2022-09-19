- Why is it that neurons sometimes align with features and sometimes don't? 
- Why do some models and tasks have many of these clean neurons, while they're vanishingly rare in others?

Superposition: When models represent more features than they have dimension. When features are sparse, superposition allows compression beyond what a linear model would do, at the cost of "interference" that requires nonlinear filtering. We offer a theory of when and why this occurs, revealing a phase diagram for superposition. We also discover that, at least in our toy model, superposition exhibits complex geometric structure.

preliminary evidence that superposition may be linked to adversarial examples and grokking

They use toy models (small ReLU networks) trained on synthetic data with sparse input features to investigate when Superposition happens. <span class="remark">why just ReLU?</span>

Key results: In our toy models, we are able to demonstrate that:
-   Superposition is a real, observed phenomenon.
-   Both monosemantic and polysemantic neurons can form.
-   At least some kinds of computation can be performed in superposition.
-   Whether features are stored in superposition is governed by a phase change. 
-   Superposition organizes features into geometric structures such as digons, triangles, pentagons, and tetrahedrons.

("but it's very unclear what to generalize to real networks")

Hypothesis: The NNs we observe in practice are in some sense noisily simulating larger, highly sparse networks.

Embed 5 feats of varying importance in 2-D over different feat. sparsity levels
- Dense (0% sparsity): 2 most imp feats represented in dedicated orth dims. Others feats not-embedded
- 80% sparsity: 4 most imp feats represented as anti-podal pairs
- 90% sparsity: all 5 feats represented and embedded as a pentagon but there's now 'positive' interference

<span class="remark">not sure what they mean by positive interference and what exactly do the directions they have shown in the images represent</span>

In our work, we often think of neural networks as having features of the input represented as directions in activation space. (I think they're calling this "linear representation hypothesis" LRH). 
We tend to think of neural network representations as being composed of features which are represented as directions.

Claiming this implies strong claims about the structure of network representations.

My interpretation of above: Whatever the feature might be (semantically meaningful or not?) there's going to be a direction in the activation space along which the variation of that feature is going to be represeted.. such that travelling along that direction would be equivlent to traversing the space of (variations in) that feature.


LRH can  be thought of as two separate properties:
- Decomposability:Network representations can be described in terms of independently understandable features.
- Linearity: Features are represented by direction.

 it's not enough for things to be decomposable: we need to be able to access the decomposition somehow. In order to do this, we need to identify the individual features within a representation. In a linear representation, this corresponds to determining which directions in activation space correspond to which independent features of the input

### Features corresponding to neurons and not

Sometimes, identifying feature directions is very easy because features seem to correspond to neurons. Why is it that we sometimes get this extremely helpful property, but in other cases don't? We hypothesize that there are really two countervailing forces driving this:
- Privileged Basis: Only some representations have a privileged basis which encourages features to align with basis directions (i.e. to correspond to neurons).
- Superposition: Linear representations can represent more features than dimensions, using a strategy we call superposition. This can be seen as neural networks simulating larger networks. This pushes features away from corresponding to neurons. (misalignmet...)

Universality - Many analogous neurons responding to the same properties can be found across networks.

Polysemantic Neurons - OTOH There are also many neurons which appear to not respond to an interpretable property of the input, and in particular, many polysemantic neurons which appear to respond to unrelated mixtures of inputs

### What are features
Our use of the term "feature" is motivated by the interpretable properties of the input we observe neurons (or word embedding directions) responding to.
 Rather than offer a single definition we're confident about, we consider three potential working definitions:

- Features as arbitrary functions: this doesn't quite seem to fit our motivations. There's something special about these features that we're observing: they seem to in some sense be fundamental abstractions for reasoning about the data, with the same features forming reliably across models.
- Features as interpretable properties: All the features we described are strikingly understandable to humans. One could try to use this for a definition: features are the presence of human understandable "concepts" in the input. But it seems important to allow for features we might not understand
- Neurons in Sufficiently Large Models:  A final approach is to define features as properties of the input which a sufficiently large neural network will reliably dedicate a neuron to representing e.g. curve detector / dog snout detector. (this approach can't define polysemantic neurons because they capture mixture of properties, so they clarify...). For interpretable properties observed in polysemantic neurons, the hope is that a sufficiently large model would dedicate a neuron to them.

We've written this paper with the final "neurons in sufficiently large models" definition in mind. But we aren't overly attached to it

### Features as directions

Linear Representation: Let's call a neural network representation linear if features correspond to directions in activation space.

One might think that a linear representation can only store as many features as it has dimensions, but it turns out this isn't the case! We'll see that the phenomenon we call superposition will allow models to store more features – potentially many more features – in linear representations.

### Privileged vs Non-privileged Bases
Even if features are encoded as directions, a natural question to ask is which directions? In some cases, it seems useful to consider the basis directions, but in others it doesn't. Why is this?

But many neural network layers are not like this. Often, something about the architecture makes the basis directions special, such as applying an activation function. This "breaks the symmetry", making those directions special, and potentially encouraging features to align with the basis dimensions. We call this a privileged basis, and call the basis directions "neurons." Often, these neurons correspond to interpretable features.

Polysemanticity is inevitable in case of Superposition. When superposition happens features can't align with the bais because model embeds more features than there are neurons.

- Almost orthogonal vectors
- Compressed sensing

Concretely, in the superposition hypothesis, features are represented as almost-orthogonal directions in the vector space of neuron outputs. Since the features are only almost-orthogonal, one feature activating looks like other features slightly activating. Tolerating this "noise" or "interference" comes at a cost. But for neural networks with highly sparse features, this cost may be outweighed by the benefit of being able to represent more features!