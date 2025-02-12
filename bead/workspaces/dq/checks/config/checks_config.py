
# === Configuration options ===

def set_config(c):
    c.input_path                   = "workspaces/dq/data/checks_data/"
    c.file_type                    = "h5"
    c.parallel_workers             = 4
    c.num_jets                     = 3
    c.num_constits                 = 15
    c.latent_space_size            = 15
    c.normalizations               = "pj_custom"
    c.invert_normalizations        = False
    c.train_size                   = 0.95
    c.model_name                   = "Conv_VAE"
    c.input_level                  = "constituent"
    c.model_init                   = "xavier"
    c.loss_function                = "MSE"
    c.optimizer                    = "adamw"
    c.epochs                       = 5
    c.lr                           = 0.001
    c.batch_size                   = 512
    c.early_stopping               = True
    c.lr_scheduler                 = True




# === Additional configuration options ===

    c.early_stopping_patience      = 100
    c.min_delta                    = 0
    c.lr_scheduler_patience        = 50
    c.reg_param                    = 0.001
    c.intermittent_model_saving    = False
    c.intermittent_saving_patience = 100
    c.activation_extraction        = False
    c.deterministic_algorithm      = False
    c.separate_model_saving        = False

