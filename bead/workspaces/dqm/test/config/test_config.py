
# === Configuration options ===

def set_config(c):
    c.workspace_name               = "dqm"
    c.project_name                 = "test"
    c.file_type                    = "h5"
    c.parallel_workers             = 16
    c.chunk_size                   = 10000
    c.num_jets                     = 3
    c.num_constits                 = 15
    c.latent_space_size            = 15
    c.normalizations               = "pj_custom"
    c.invert_normalizations        = False
    c.train_size                   = 0.95
    c.model_name                   = "Planar_ConvVAE"
    c.input_level                  = "constituent"
    c.input_features               = "4momentum_btag"
    c.model_init                   = "xavier"
    c.loss_function                = "VAELoss"
    c.optimizer                    = "adamw"
    c.epochs                       = 2
    c.lr                           = 0.001
    c.batch_size                   = 2
    c.early_stopping               = True
    c.lr_scheduler                 = True
    c.latent_space_plot_style      = "umap"
    c.subsample_plot               = False

# === EFP (Energy-Flow Polynomial) configuration ===
    c.efp_mode                     = False
    c.efp_nmax                     = 5
    c.efp_dmax                     = 6
    c.efp_beta                     = 1.0
    c.efp_measure                  = "hadr"
    c.efp_normed                   = True
    c.efp_include_composites       = False
    c.efp_standardize_meanvar      = True
    c.efp_n_jobs                   = 4

# === NT-Xent loss configuration ===
    c.ntxent_sigma                 = 0.1
    c.lr_scheduler_patience        = 30
    c.reg_param                    = 0.001
    c.intermittent_model_saving    = True
    c.intermittent_saving_patience = 100
    c.subsample_size               = 300000
    c.contrastive_temperature      = 0.07
    c.contrastive_weight           = 0.005
    c.overlay_roc                  = False
    c.overlay_roc_projects         = ["workspace_name/project_name"]
    c.overlay_roc_save_location    = "overlay_roc"
    c.overlay_roc_filename         = "combined_roc.pdf"


# === Test loss histogram configuration ===
    c.plot_test_loss_histogram     = False  # Set to True to include test loss histogram in standard plotting
    c.plot_only_test_loss_histogram = True  # Set to True to skip all other plots and only generate test loss histogram
    c.test_loss_histogram_component = "loss_test"  # Which loss component to plot (e.g., "loss_test", "kl_test", "reco_test")

# === Parameter annealing configuration ===
    c.annealing_params = {
        "reg_param": {
            "strategy": "TRIGGER_BASED",
            "values": [0.001, 0.005, 0.01],
            "trigger_source": "early_stopper_third_patience",
            "current_index": 0
        },
        "contrastive_weight": {
            "strategy": "TRIGGER_BASED",
            "values": [0, 0.005, 0.01, 0.02, 0.03],
            "trigger_source": "early_stopper_half_patience",
            "current_index": 0
        }
    }
# === Additional configuration options ===

    c.use_ddp                      = False
    c.use_amp                      = False
    c.early_stopping_patience      = 100
    c.min_delta                    = 0
    c.activation_extraction        = False
    c.deterministic_algorithm      = False
    c.separate_model_saving        = False

    c.plot_roc_per_signal          = True
    c.skip_to_roc                  = False
    c.subsample_plot               = False
    c.use_amp                      = False
    c.use_ddp                      = False
