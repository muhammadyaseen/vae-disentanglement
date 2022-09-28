# Strings
LOSS = 'loss'
ACCURACY = 'acc'
ITERATION = 'iteration'
WANDB_NAME = 'disentanglement'
INPUT_IMAGE = 'input_image'
RECON_IMAGE = 'recon_image'
RECON = 'recon'
FIXED = 'fixed'
SQUARE = 'square'
ELLIPSE = 'ellipse'
HEART = 'heart'
TRAVERSE = 'traverse'
RANDOM = 'random'
TEMP = 'tmp'
GIF = 'gif'
JPG = 'jpg'
FACTORVAE = 'FactorVAE'
DIPVAE_I = 'DIPVAEI'
DIPVAE_II = 'DIPVAEII'
BetaTCVAE = 'BetaTCVAE'
INFOVAE = 'InfoVAE'
TOTAL_VAE = 'total_vae'
TOTAL_LOSS = 'loss' # required for PyTorch Lightining loss funcs
KLD_LOSS = 'kld_loss'
TOTAL_VAE_EPOCH = 'total_vae_epoch'
LEARNING_RATE = 'learning_rate'
AUX_CLASSIFICATION = 'aux_classification'
COVARIANCE_LOSS = 'covariance_loss'

# Algorithms
ALGS = ('AE', 'VAE', 'BetaVAE', 'CVAE', 'IFCVAE', 'FC_VAE', 'LadderVAE', 
        'ConceptStructuredVAE', 'CSVAE_ResidualDistParameterization','CSVAE_Toy', 'GNNBasedConceptStructuredVAE',
        'LatentToImage', 'LatentNN_CSVAE')

LOSS_TERMS = (FACTORVAE, DIPVAE_I, DIPVAE_II, BetaTCVAE, INFOVAE, AUX_CLASSIFICATION, COVARIANCE_LOSS)

# Datasets
DATASETS = ('dsprites_cond', 'polynomial',
            'dsprites_noshape', 'color_dsprites', 'noisy_dsprites', 'scream_dsprites',
            'smallnorb', 'cars3d', 'shapes3d',
            'mpi3d_toy', 'mpi3d_realistic', 'mpi3d_real')

# Datasets for which we have loaders in common.known_datasets and that aren't DisentLibDatasets
KNOWN_DATASETS = (
    'celeba', 'dsprites_full', 'onedim','threeshapes','threeshapesnoisy', 
    'continum', 'dsprites_colored', 'dsprites_cond', 'polynomial', 
    'toydata', 'flow', 'pendulum', 'pendulum_switch'
)
# Datasets for which we have loaders in common.known_datasets and that ARE DisentLibDatasets
KNOWN_DISENTLIB_DATASETS = ('dsprites_correlated',)

DEFAULT_DATASET = DATASETS[-2]  # mpi3d_realistic
TEST_DATASETS = DATASETS[0:2]  # celebA, dsprites_full

# Architectures
DISCRIMINATORS = ('SimpleDiscriminator', 'SimpleDiscriminatorConv64')
TILERS = ('MultiTo2DChannel',)
DECODERS = ('SimpleConv64', 'ShallowLinear', 'DeepLinear','SimpleFCNNDencoder',
            'SimpleConv64CommAss', 'SmallDecoder','SmallFCDecoder')
ENCODERS = ('SimpleConv64', 'SimpleGaussianConv64', 'PadlessConv64', 'PadlessGaussianConv64',
            'ShallowGaussianLinear', 'DeepGaussianLinear','SimpleFCNNEncoder', 
            'SimpleGaussianConv64CommAss', 'SmallEncoder', 'SmallFCEncoder')

# Evaluation Metrics
EVALUATION_METRICS = ('dci', 'factor_vae_metric', 'sap_score', 'mig', 'irs', 'beta_vae_sklearn')

# Schedulers
LR_SCHEDULERS = ('ReduceLROnPlateau', 'StepLR', 'MultiStepLR', 'ExponentialLR',
                 'CosineAnnealingLR', 'CyclicLR', 'LambdaLR')
SCHEDULERS = ('LinearScheduler', )
