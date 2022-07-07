import torch
import tensorflow as tf

# Pytorch_VAE - AntixK - BeaVAE [ADD]
# https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py#L141
kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
# rewrite
kld_loss = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(dim = 1).mean(dim = 0)

# take minus sign inside
kld_loss = 0.5 * (- 1 - log_var + mu ** 2 + log_var.exp()).sum(dim = 1).mean(dim = 0)

# betaVAE Google disentlib [ADD]
# D:\Saarbrucken\EDA_Research\disentanglement_lib\disentanglement_lib\methods\unsupervised\vae.py#118
kld = tf.reduce_mean(
      0.5 * tf.reduce_sum(
          tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1, [1]),
      name="kl_loss")

# prev mine [ADD]
kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
kld = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(1).mean()

# current mine
kld = -0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(1).mean()

#1-konny github [ADD]
# https://github.com/1Konny/Beta-VAE/blob/977a1ece88e190dd8a556b4e2efb665f759d0772/solver.py#L44
klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean(0)
# take minus sign inside
klds = 0.5 * (- 1 - logvar + mu.pow(2) + logvar.exp()).sum(1).mean(0)

# benchmark var [ADD]
# https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/beta_vae/beta_vae_model.py#L101
KLD = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(-1).mean(0)

# Pytorch example [ADD]
# https://github.com/pytorch/examples/blob/main/vae/main.py#L80
KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

