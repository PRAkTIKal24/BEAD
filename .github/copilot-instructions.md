# BEAD Copilot Instructions

## Project Overview
BEAD (Background Enrichment for Anomaly Detection) is a PyTorch-based deep learning framework for anomaly detection in high-energy physics (HEP) data. It uses unsupervised latent variable models (VAEs, autoencoders, normalizing flows) to learn enriched background representations and detect new physics signals.

## Architecture & Key Components

### Data Pipeline
- **CSV → H5/NPY → PyTorch Tensors**: Data flows through three stages:
  1. **Conversion** ([conversion.py](bead/src/utils/conversion.py)): Parses CSV files into HDF5 with structured arrays for event-level, jet-level, and constituent-level data
  2. **Processing** ([data_processing.py](bead/src/utils/data_processing.py)): Loads H5, normalizes, selects top-N jets/constituents, converts to tensors
  3. **Tensors**: Saved as `.pt` files for training/inference efficiency

### CLI Architecture ([ggl.py](bead/src/utils/ggl.py))
- **Control Center**: Single entry point for all workflows
- **Workspace/Project Structure**: `bead/workspaces/{workspace_name}/{project_name}/`
  - Shared workspace data: `data/csv`, `data/h5`, `data/npy` (reused across projects)
  - Per-project configs: `config/{project_name}_config.py`
  - Per-project outputs: `output/models`, `output/results`, `output/plots`
- **Key Functions**: `create_new_project()` → `convert_csv()` → `prepare_inputs()` → `run_training()` → `run_inference()` → `run_plots()`
- **Chain Mode**: Underscore-separated options like `"convertcsv_prepareinputs_train_detect"` execute multiple stages sequentially

### Model Architecture ([models.py](bead/src/models/models.py), [flows.py](bead/src/models/flows.py), [layers.py](bead/src/models/layers.py))
- **Base Models**: AE, ConvAE, ConvVAE, Dirichlet_ConvVAE
- **Flow Variants**: Planar, OrthogonalSylvester, HouseholderSylvester, TriangularSylvester, IAF, ConvFlow, NSFAR
- **Naming Convention**: `{FlowType}_{BaseModel}` (e.g., `Planar_ConvVAE`)
- **Transformer Integration**: Optional transformer components for enhanced feature learning

### Loss Functions ([loss.py](bead/src/utils/loss.py))
- **Hierarchy**: BaseLoss → specific losses (ReconstructionLoss, KLDivergenceLoss, ContrastiveLoss)
- **Compound Losses**: VAELoss (reco + KL), VAEFlowLoss, DVAELoss (Dirichlet), with optional regularization (L1, L2, EMD)
- **Contrastive Losses**: SupCon and NT-xEnt (CRVSTAL framework) for leveraging generator labels

### Hyperparameter Annealing ([annealing.py](bead/src/utils/annealing.py))
- **Three Strategies**: CONSTANT_PACE, TRIGGER_BASED, SCHEDULED
- **DDP Broadcasting**: Rank 0 makes decisions, broadcasts to other ranks
- **Common Annealed Params**: `contrastive_temperature`, `reg_param`, `contrastive_weight`
- **Config Usage**:
  ```python
  c.annealing_params = {
      "param_name": {
          "strategy": "CONSTANT_PACE",
          "start_value": 0.1, "end_value": 0.01, "total_steps": 10
      }
  }
  ```

## Critical Workflows

### Training Pipeline ([training.py](bead/src/trainers/training.py))
1. **DDP Setup**: Initialize `torch.distributed` if `config.use_ddp=True` and multi-GPU available
2. **fit()**: Single epoch training with mixed precision (AMP) and gradient clipping
3. **validate()**: Evaluation on validation set with early stopping
4. **Distributed Synchronization**: 
   - Only rank 0 prints progress
   - All ranks have synchronized loss metrics
   - Early stopping decision from rank 0 broadcasted to all

### Inference Pipeline ([inference.py](bead/src/trainers/inference.py))
- Loads trained model from `output/models/`
- Evaluates on test set, saves loss/anomaly scores to `output/results/`
- Supports batch processing for large datasets

### Plotting & Analysis ([plotting.py](bead/src/utils/plotting.py), [statistical_plotting.py](bead/src/utils/statistical_plotting.py))
- **GPU Acceleration**: Uses cuML (RAPIDS) for PCA/t-SNE/UMAP when available, falls back to CPU
- **ROC Curves**: Standard per-signal and overlay modes for multi-project comparison
- **Dimensionality Reduction**: Handles t-SNE, UMAP, TRIMAP, PCA (both CPU and GPU)
- **Overlay Feature**: Compare ROCs across workspaces via `config.overlay_roc` and `config.overlay_roc_projects`

## Project-Specific Patterns

### Configuration Design
- **Config Class**: Attributes set in `set_config(c)` function in `{project_name}_config.py`
- **Data Levels**: `"constituent"`, `"jet"`, or `"event"`
- **Input Features**: `"4momentum"`, `"4momentum_btag"`, `"efp"` (Energy Flow Polynomials)
- **Normalizations**: `"pj_custom"`, `"standard"`, `"robust"`
- **Workspace Isolation**: Different workspaces for different input data; same workspace for parameter/model variations

### Energy Flow Polynomials (EFP)
- **Integration** ([efp_utils.py](bead/src/utils/efp_utils.md)): Uses EnergyFlow package for jet polynomial features
- **Config**: Set `c.input_features = "efp"` and configure EFP set via `c.efp_config`
- **Caching**: Precomputes and caches EFP datasets for faster training

### Testing
- **Test Location**: `tests/unit/` (pytest-based)
- **Key Tests**: Config creation, annealing, EFP embedding, GPU plotting, loss functions
- **Run**: `uv run pytest tests/unit/ -v`
- **Coverage**: Configured in `pyproject.toml` with `pytest-cov`

### Version Management
- **Package Manager**: uv (not pip) — handles dependency resolution and environment isolation
- **Python Version**: >=3.10, <3.13
- **Key Dependencies**: PyTorch (GPU/CPU variants auto-selected), scikit-learn, h5py, numba, dask, trimap
- **Optional Groups**: `[gpu]` (cuML, cudf), `[viz]` (umap-learn), `[test]` (pytest, pytest-cov)
- **Install**: `uv pip install -e .` or `uv sync`

### Multi-GPU Training (DDP)
- **Activation**: Set `config.use_ddp=True` in config file
- **Launch**: Use `torchrun --standalone --nnodes=N --nproc_per_node=M -m bead.bead [args]`
- **Environment**: Manually create venv with `uv pip install -e <VENV_PATH>` for HPC compatibility
- **Synchronization**: AnnealingManager broadcasts rank 0 decisions; DistributedSampler handles data sharding
- **Limitation**: Currently tested only with homogeneous GPU setups (same GPU type)

### File Naming Conventions
- **CSV Flags**: Files must start with `bkg_train`, `bkg_test`, or `sig_test` for auto-concatenation
- **Output Files**: Results saved as `.npy` arrays (loss, labels, metrics) with level suffix (e.g., `loss_test.npy`, `test_{level}_label.npy`)
- **Model Files**: Named as `{project_name}_model_state.pt`

## Common Development Tasks

### Adding a New Model
1. Inherit from `nn.Module` in [models.py](bead/src/models/models.py)
2. Implement `forward()`, `encode()`, `decode()`
3. For VAE: add `mu`, `logvar` outputs from encoder
4. For Flows: integrate flow classes from [flows.py](bead/src/models/flows.py)
5. Register model in config under `c.model_name = "YourModelName"`

### Adding a New Loss Function
1. Create class inheriting `BaseLoss` in [loss.py](bead/src/utils/loss.py)
2. Implement `calculate()` method returning tuple of loss components
3. Support DDP broadcasting of losses via `torch.distributed`
4. Register in config under `c.loss_function = "YourLossName"`

### Modifying Data Processing
- **Normalization**: Add strategy to [normalization.py](bead/src/utils/normalization.py), register in config
- **Feature Selection**: Modify `select_top_jets_and_constituents()` in [data_processing.py](bead/src/utils/data_processing.py)
- **New Input Level**: Requires changes in conversion, processing, and all model forward passes

## Debugging & Diagnostics

### Verbose Mode
- Add `-v` flag to any command: `uv run bead -m train -p WS PROJ -v`
- Enables detailed progress output and shape tracking

### Diagnostics Mode
- Run `uv run bead -m diagnostics -p WS PROJ`
- Generates profiling metrics for CPU/GPU usage optimization ([diagnostics.py](bead/src/utils/diagnostics.py))

### Common Issues
- **GPU out of memory**: Reduce `config.batch_size` or increase `config.num_workers` for better pipeline efficiency
- **DDP hanging**: Ensure all ranks receive matching model/optimizer initialization; check rank synchronization points
- **Data mismatch**: Verify CSV files are in correct location with correct flags before conversion
- **Import errors**: Run `uv sync` to ensure environment is synchronized

## Entry Points & CLI
- **Main Script**: [bead/bead.py](bead/bead.py) → `main()` function
- **CLI Hub**: [bead/src/utils/ggl.py](bead/src/utils/ggl.py) → `get_arguments()`, `run_full_chain()`
- **Command Format**: `uv run bead -m {mode} -p {workspace} {project} [-o options] [-v]`
