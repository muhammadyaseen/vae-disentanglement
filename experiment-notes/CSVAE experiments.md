- [ ] Implement classification heads and think of a more general strategy to localize information in units
- [ ] Debug Dying units / mu problem - Sparsity or Implementation Bug ?
- [ ] Run both nets for 5 epochs to check if correct plots are being drawn
- [ ] Histogram and Scalar doesn't show final layer data
- [ ] Is `detach()` that I'm doing in `_top_down_pass()` the reason for zero entries?



### Experiments with Increased Capacity 
I did two sets of experiments on DSprites dataset
1. IC CSVAE without KLD loss scheduling
	In this case the same problems persisted. Loss doesn't change much and the activations are close to zero. Reconstructions are very bad, just a dull gray square-ish blob in the center
	
	![[iccsvae-e1.png]]
2. IC CSVAE with KLD loss scheduling
	Same behaviour as above.
	![[iccsvae-e2.png]]

![[csvae_dsprites_after_ic_recons.png]]
KLD loss is zero in loss scheduling, because we only introduce KLD loss after K=10 epochs. Even then, the reconstruction loss - which is the only effective loss as this point - doesn't change much. This behaviour or recon loss is similar to non-IC case. 
This makes *some* sense. If the layers only output $\mu=0$ and $\log \sum=0$ then we will have zero KLD loss. In top layer because it is being regularized to $\mathcal{N}(0,I)$ and in 2nd layer because (...because why? - Actually I don't know this - why is 2nd layer doing bad ?) I think for this I need to check the corresponding activations in `Bottom Up` layers.

Another observation is that the <u>Recon loss starts at a higher value</u> compared to Previous i.e. Before Capacity Increase (~40 vs  ~140) and the mu activations are all almost zero eventhough because of scheduling there is no regularization pressure. So it basically got even worse.
The two phenomenon go together because if activations are zero / non-informative than we of course can't get a good reconstruction. So I should first investigate why all the activations are zero. I had already noticed these zeros when I was verifying structure of CSVAE. I should now look deeper into it.


### Increased Capacity Implementation

I have restructured CSVAE and implemented it in such a way that now we can have multiple dense intermediate connections b/w parents and children units.
Main motivation for this re-implementation was to allow for more capacity in intermediate layers because it seemed that network didn't learn enough / was getting bottle-necked because of sparse connections and low units

I need to plot the latent activations or their histogram because <u>it seems in the test runs that many dimensions simply go to zero</u> . E.g. only 2 out of 5 dims get non-zero entries. I'm not yet sure if this is expected because of sparsity or some implementation bug.
- [ ] Sparsity or Implementation Bug ?

I'm also not sure on what is the best way to visualize / interpret these dense `Intermediate` layers. Earlier when there was only a single hidden layer, we could plot the weight matrix. But now ... it isn't that simple.


### Before Expanded Capacity

CS-VAE result on DSprites with corr=0.2 epochs=30 | Ran on 31-05-2022, Stored in as`train-logs/CS_VAE_dsprites_corr02_before_IC`

Observations:
	- KLD loss for latents lower in the hierarchy is going to zero and reconstruction is very bad. White-ish circular blob in the middle with some size variation that is moslty similar to the actual sprite size. So i'd say that the network is capturing the Color and Size, but not anything else.
	-  KLD loss doesn't change much over ~30 epochs, staying b/w 7.5 and 6.8 for layer 0 and b/w 0.15 and 0.04 for layer 1
	-  Almost all the $\mu$ components in all latent layers are <u>very close to zero</u> from above (small positive numbers)
	- Losses seem to be behaving erratically, going up and down.
	- Similar trend for reconstruction loss, it doesn't change much after first 2 epochs and stays b/w 35 - 31
![[csvae_before_ic_dsprites_loss.png]]
![[csvae_before_ic_dsprites_recons.png]]
Note: I don't recall if in this network I used a dense root note or a single unit one. 
**Update:** It doesn't matter because DSprites does not have a single root node to begin with. There are 4 latents in the top layer. This was also confirmed by looking at Mu plots

Take aways:
	- It seems that training is stuck, without any improvement. Could be because of <u>Low capacity in the intermediate layers</u> ?
	- I think that current architecture doesn't have enough capacity to carry info to the decoder. The `DAGInteractionLayer` causes a severe information bottleneck. I should try increasing `interm_unit_dim` , then possibly including more intermediate layers ?
	- In the Bottom Up networks i'm using `SimpleFCNNEncoder` which is a network with only 1 hidden layer, I can also try out increasing its depth.

Making units relevant to the labels:
	- Another question is... I have to somehow make the units relevant to the labels. There is an easy way to do this, but I have to think + implement it.
	- Need to think about how it would be implemented - in a general way
 
What is the correct way to visualize $\mu$ components in `train_step_end()` ? 
Right now I'm averaging the the value of dimension k for each example in a batch so I'm plotting $$\mu_k = \frac{1}{B} \sum_{b=1}^B \mu_b[k]$$, but I think it would naturally tend towards 0 ? Should I instead be plotting histogram of mean components per epoch ? **Update:** Added histogram plot

In bottom-up network `SimpleFCNNEncoder` we are using `Tanh()` activation and then a linear layer, this can output -ve values. Since we use half of the output layer as sigma, this isn't really valid. **Update:** Removed `Tanh` and added `RelU`

