- We need to make sure that both Recon term and KL term are in same units. This can easily be achieved for datasets where Recon is a kind of CrossEntropy.
- In case of color images we might need to estimate some kind of lower/upper bound to this.
- Then the weightage can be expressed as a kind of "trade value" e.g. we can buy X bits of Recon, if we spend Y bits of KL div . This might give us an exchange rate that can be used to calibrate the weightage.
- For synth dataset the entropy of latent distribution is easy to estimate but for datasets like CelebA it is more tricky, again we might need some kind of bounds on entropy of latent randoms

For ex in `Dsprites` we can estimate the entropy of latent disributions as follows:
$H = \sum_{i \in \text{factors}} H_i$ where $H_i$ is simply given by $\log(N_i)$ and $N_i$ is the unique number of latent values the factor can take. For the 6 factors this comes out to be: $$ \log 1 + \log 3 + \log 6 + \log 40 + \log 32 + \log 32 $$$$ = \log(1 \times 3 \times 6 \times 40 \times 32 \times 32)$$$$ = \log(737280) \approx 19.50 \text{bits}$$
<span class="remark">I should check the network that had very good reconstruction if this bit capacity holds here. it would be an expected but a really cool find if it does</span>

We could do this because all the factors were independent in DAG situations we won't be able to do it this easily.

Need to think also about correlations etc in the Observed domain. I think cross ent treats every pixel independently

In this way, we actually do know the minimum capacity the channel needs to express our latents using the true distribution.


#### FixedWeightsDsprites experiments investigation

| Variable      | Value |
| ----------- | ----------- |
| $w_{recon}$   | $0.8$        |
| $w_{kld}$   | $0.2$        |
| Actual $L_{recon}$   | $0.8$        |
| Weighted $L_{recon}$   | $0.8$        |
| Actual $L_{kld}$   | $0.8$        |
| Weighted $L_{kld}$   | $0.8$        |



kldiv is in nats, need to be sure that recon is also in nats
