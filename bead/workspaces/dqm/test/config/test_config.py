
# === Configuration options ===

def set_config(c):
    c.workspace_name               = "dqm"
    c.project_name                 = "test"
    c.file_type                    = "h5"
    c.parallel_workers             = 8
    c.chunk_size                   = 10000
    c.num_jets                     = 3
    c.num_constits                 = 15
    c.latent_space_size            = 15
    c.normalizations               = "pj_custom"
    c.invert_normalizations        = False
    c.train_size                   = 0.95
    c.model_name                   = "ConvVAE"
    c.input_level                  = "constituent"
    c.input_features               = "4momentum_btag"
    c.model_init                   = "xavier"
    c.loss_function                = "VAEFlowLoss"
    c.optimizer                    = "adamw"
    c.epochs                       = 4
    c.lr                           = 0.001
    c.batch_size                   = 512
    c.early_stopping               = True
    c.lr_scheduler                 = True
    c.latent_space_plot_style      = "trimap"




# === Additional configuration options ===

    c.early_stopping_patience      = 100
    c.min_delta                    = 0
    c.lr_scheduler_patience        = 50
    c.reg_param                    = 0.001
    c.intermittent_model_saving    = True
    c.intermittent_saving_patience = 2
    c.activation_extraction        = False
    c.deterministic_algorithm      = False
    c.separate_model_saving        = False

