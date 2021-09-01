

### Usage
```
$ cd disent-vars
$ python run.py -c configs/<config-file-name.yaml>
```
**Config file template**
```yaml
model_params:
  name: "<name of VAE model>"
  in_channels: 3
  latent_dim: 
    .         # Other parameters required by the model
    .
    .

exp_params:
  data_path: "<path to the celebA dataset>"
  img_size: 64    # Models are designed to work for this size
  batch_size: 64  # Better to have a square number
  LR: 0.005
  weight_decay:
    .         # Other arguments required for training, like scheduler etc.
    .
    .

trainer_params:
  gpus: 1         
  max_nb_epochs: 50
  gradient_clip_val: 1.5
    .
    .
    .

logging_params:
  save_dir: "logs/"
  name: "<experiment name>"
  manual_seed: 
```

**View TensorBoard Logs**
```
$ cd logs/<experiment name>/version_<the version you want>
$ tensorboard --logdir tf
```

----
<h2 align="center">
  <b>Results</b><br>
</h2>


| Model                                                                  | Paper                                            |
|------------------------------------------------------------------------|--------------------------------------------------|
| VAE ([Code][vae_code], [Config][vae_config])                           |[Link](https://arxiv.org/abs/1312.6114)           |  
| Conditional VAE ([Code][cvae_code], [Config][cvae_config])             |[Link](https://openreview.net/forum?id=rJWXGDWd-H)|  
| WAE - MMD (RBF Kernel) ([Code][wae_code], [Config][wae_rbf_config])    |[Link](https://arxiv.org/abs/1711.01558)          |  
| WAE - MMD (IMQ Kernel) ([Code][wae_code], [Config][wae_imq_config])    |[Link](https://arxiv.org/abs/1711.01558)          |  
| Beta-VAE ([Code][bvae_code], [Config][bbvae_config])                   |[Link](https://openreview.net/forum?id=Sy2fzU9gl) |  
| Disentangled Beta-VAE ([Code][bvae_code], [Config][bhvae_config])      |[Link](https://arxiv.org/abs/1804.03599)          |  
| Beta-TC-VAE ([Code][btcvae_code], [Config][btcvae_config])             |[Link](https://arxiv.org/abs/1802.04942)          |  
| IWAE (*K = 5*) ([Code][iwae_code], [Config][iwae_config])              |[Link](https://arxiv.org/abs/1509.00519)          |  
| MIWAE (*K = 5, M = 3*) ([Code][miwae_code], [Config][miwae_config])    |[Link](https://arxiv.org/abs/1802.04537)          |  
| DFCVAE   ([Code][dfcvae_code], [Config][dfcvae_config])                |[Link](https://arxiv.org/abs/1610.00291)          |  
| MSSIM VAE    ([Code][mssimvae_code], [Config][mssimvae_config])        |[Link](https://arxiv.org/abs/1511.06409)          |  
| Categorical VAE   ([Code][catvae_code], [Config][catvae_config])       |[Link](https://arxiv.org/abs/1611.01144)          |  
| Joint VAE ([Code][jointvae_code], [Config][jointvae_config])           |[Link](https://arxiv.org/abs/1804.00104)          |  
| Info VAE   ([Code][infovae_code], [Config][infovae_config])            |[Link](https://arxiv.org/abs/1706.02262)          |  
| LogCosh VAE   ([Code][logcoshvae_code], [Config][logcoshvae_config])   |[Link](https://openreview.net/forum?id=rkglvsC9Ym)|  
| SWAE (200 Projections) ([Code][swae_code], [Config][swae_config])      |[Link](https://arxiv.org/abs/1804.01947)          |  
| VQ-VAE (*K = 512, D = 64*) ([Code][vqvae_code], [Config][vqvae_config])|[Link](https://arxiv.org/abs/1711.00937)          |  
| DIP VAE ([Code][dipvae_code], [Config][dipvae_config])                 |[Link](https://arxiv.org/abs/1711.00848)          | 

<!-- | Gamma VAE             |[Link](https://arxiv.org/abs/1610.05683)          |


[vae_code]: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
[cvae_code]: https://github.com/AntixK/PyTorch-VAE/blob/master/models/cvae.py
[bvae_code]: https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py
[btcvae_code]: https://github.com/AntixK/PyTorch-VAE/blob/master/models/betatc_vae.py
[wae_code]: https://github.com/AntixK/PyTorch-VAE/blob/master/models/wae_mmd.py
[iwae_code]: https://github.com/AntixK/PyTorch-VAE/blob/master/models/iwae.py
[miwae_code]: https://github.com/AntixK/PyTorch-VAE/blob/master/models/miwae.py
[swae_code]: https://github.com/AntixK/PyTorch-VAE/blob/master/models/swae.py
[jointvae_code]: https://github.com/AntixK/PyTorch-VAE/blob/master/models/joint_vae.py
[dfcvae_code]: https://github.com/AntixK/PyTorch-VAE/blob/master/models/dfcvae.py
[mssimvae_code]: https://github.com/AntixK/PyTorch-VAE/blob/master/models/mssim_vae.py
[logcoshvae_code]: https://github.com/AntixK/PyTorch-VAE/blob/master/models/logcosh_vae.py
[catvae_code]: https://github.com/AntixK/PyTorch-VAE/blob/master/models/cat_vae.py
[infovae_code]: https://github.com/AntixK/PyTorch-VAE/blob/master/models/info_vae.py
[vqvae_code]: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
[dipvae_code]: https://github.com/AntixK/PyTorch-VAE/blob/master/models/dip_vae.py

[vae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/vae.yaml
[cvae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/cvae.yaml
[bbvae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/bbvae.yaml
[bhvae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/bhvae.yaml
[btcvae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/betatc_vae.yaml
[wae_rbf_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/wae_mmd_rbf.yaml
[wae_imq_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/wae_mmd_imq.yaml
[iwae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/iwae.yaml
[miwae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/miwae.yaml
[swae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/swae.yaml
[jointvae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/joint_vae.yaml
[dfcvae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/dfc_vae.yaml
[mssimvae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/mssim_vae.yaml
[logcoshvae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/logcosh_vae.yaml
[catvae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/cat_vae.yaml
[infovae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/infovae.yaml
[vqvae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/vq_vae.yaml
[dipvae_config]: https://github.com/AntixK/PyTorch-VAE/blob/master/configs/dip_vae.yaml