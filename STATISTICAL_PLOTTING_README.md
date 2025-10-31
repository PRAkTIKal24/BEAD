# Statistical Plotting for ROC Analysis

This document describes the new statistical plotting functionality added to BEAD for analyzing model performance variation across different signal samples.

## Overview

The statistical plotting functions parse terminal output from the `roc_per_signal` mode and generate visualizations showing the statistical variation of different models across each available signal sample individually. This provides insights into model performance consistency and highlights differences between models.

## Features

### Plot Types

1. **Box Plots** (`*_box_plots.pdf`)
   - Show statistical summaries (median, quartiles, outliers) for each model
   - Each metric (AUC, TPR at different FPR levels) has its own subplot
   - Models are displayed along the x-axis with right-aligned labels

2. **Violin Plots** (`*_violin_plots.pdf`)
   - Show the full distribution shape for each model's performance
   - Includes density estimation showing probability distribution
   - Reveals multi-modal distributions and distribution skewness

3. **Combined Plots** (`*_combined_plots.pdf`)
   - Overlay box plots on top of violin plots
   - Violin plots provide background distribution information
   - Box plots provide statistical summary overlays
   - Best of both visualization types

4. **Parameterized Violin Plots** (`*_parameterized_violin_plots.pdf`)
   - Enhanced violin plots that show both overall performance and signal parameter effects
   - Each model's violin shows the overall distribution (semi-transparent background)
   - Colored scatter points indicate individual signal performance, color-coded by parameters
   - **Mass-based colors**: Each mediator mass gets a distinct color from Set1 colormap
   - **R_invisible transparency**: Different alpha values for visual distinction
   - Legend shows all mediator mass and R_invisible combinations
   - Reveals how model performance varies across different signal characteristics

5. **Parameterized Box Plots** (`*_parameterized_box_plots.pdf`) - **NEW!**
   - Box plots with signal parameter color-coding
   - Shows statistical summaries grouped by signal parameters
   - Same enhanced color scheme as parameterized violin plots
   - Ideal for comparing median performance across signal characteristics

6. **Parameterized Combined Plots** (`*_parameterized_combined_plots.pdf`) - **NEW!**
   - Combines parameterized violin and box plots
   - Full distribution shapes with statistical summary overlays
   - Complete visualization of both overall and parameter-specific performance
   - Most comprehensive analysis option

### Metrics Analyzed

- **AUC**: Area Under the ROC Curve
- **TPR at FPR 1e-4**: True Positive Rate at False Positive Rate 10^-4
- **TPR at FPR 1e-3**: True Positive Rate at False Positive Rate 10^-3
- **TPR at FPR 1e-2**: True Positive Rate at False Positive Rate 10^-2

### Color Schemes

#### Enhanced Parameter-Based Color Scheme
For parameterized plots (violin, box, and combined), an advanced color scheme is used:

- **Mass-based colors**: Each mediator mass gets a specific color:
  - **1000 GeV**: Red
  - **2000 GeV**: Yellow
  - **3000 GeV**: Green
  - **4000 GeV**: Blue
  - **5000 GeV**: Violet
- **R_invisible transparency**: Different alpha (transparency) values distinguish R_invisible values:
  - R_invisible = 0.25: α = 0.9 (most opaque)
  - R_invisible = 0.5: α = 0.7 (medium transparency)
  - R_invisible = 0.75: α = 0.5 (most transparent)
- This dual encoding allows easy identification of both mass and R_invisible effects

#### CSF Results Workspace (Basic Plots)
- `convvae`: Blue
- `convvae_planar`: Orange
- `convvae_house`: Green
- `ntx_convvae`: Red
- `dvae`: Violet
- `convvae_sc`: Brown
- `convvae_house_sc_anneal`: Pink

#### Other Workspaces (Basic Plots)
- Uses matplotlib's default tab10 colormap for consistent color assignment

## Usage

### Method 1: Standalone Script

```bash
# Basic usage
uv run python generate_statistical_plots.py roc_output.txt

# With custom output directory
uv run python generate_statistical_plots.py roc_output.txt ./my_plots/

# With verbose output
uv run python generate_statistical_plots.py roc_output.txt -v

# Skip 5000 GeV signals (useful for focused analysis on lower masses)
uv run python generate_statistical_plots.py roc_output.txt ./plots_no_5000/ --skip-5000 -v
```

### Method 2: Python Import

```python
from bead.src.utils.statistical_plotting import generate_statistical_plots_from_roc_output

# Generate all plots
generate_statistical_plots_from_roc_output(
    output_file_path="roc_output.txt",
    save_dir="./statistical_plots/",
    verbose=True
)

# Generate plots without 5000 GeV signals
generate_statistical_plots_from_roc_output(
    output_file_path="roc_output.txt",
    save_dir="./statistical_plots_no_5000/",
    verbose=True,
    skip_5000=True
)
```

### Method 3: Individual Plot Types

```python
from bead.src.utils.statistical_plotting import (
    parse_roc_output,
    create_box_plots,
    create_violin_plots,
    create_combined_box_violin_plots,
    create_parameterized_violin_plots,
    create_parameterized_box_plots,
    create_parameterized_combined_plots
)

# Parse the data
parsed_data = parse_roc_output("roc_output.txt", verbose=True)

# Generate specific plot types
create_box_plots(parsed_data, "./plots/", verbose=True)
create_violin_plots(parsed_data, "./plots/", verbose=True)
create_combined_box_violin_plots(parsed_data, "./plots/", verbose=True)

# Generate parameterized plots with enhanced color schemes
create_parameterized_violin_plots(parsed_data, "./plots/", verbose=True)
create_parameterized_box_plots(parsed_data, "./plots/", verbose=True)  # NEW!
create_parameterized_combined_plots(parsed_data, "./plots/", verbose=True)  # NEW!

# Generate plots without 5000 GeV signals
create_parameterized_violin_plots(parsed_data, "./plots_no_5000/", verbose=True, skip_5000=True)
create_parameterized_box_plots(parsed_data, "./plots_no_5000/", verbose=True, skip_5000=True)
create_parameterized_combined_plots(parsed_data, "./plots_no_5000/", verbose=True, skip_5000=True)
```

## Input Format

The functions expect terminal output from BEAD's `roc_per_signal` mode. The expected format includes:

```
Processing signal file: sneaky1000R025 (indices 1407905:1419763)
  sneaky1000R025 - LOSS AUC: 0.6364
    TPR at FPR 1.0e-04: 0.0027
    TPR at FPR 1.0e-03: 0.0192
    TPR at FPR 1.0e-02: 0.0712
    TPR at FPR 1.0e-01: 0.2576
Saved per-signal ROC plot: bead/workspaces/csf_results/convvae/output/plots/loss/roc_sneaky1000R025.pdf
```

## Output Files

For each workspace found in the ROC output, six PDF files are generated:

- `{workspace_name}_box_plots.pdf`
- `{workspace_name}_violin_plots.pdf`
- `{workspace_name}_combined_plots.pdf`
- `{workspace_name}_parameterized_violin_plots.pdf`
- `{workspace_name}_parameterized_box_plots.pdf` - **NEW!**
- `{workspace_name}_parameterized_combined_plots.pdf` - **NEW!**

The parameterized plots use enhanced color schemes to show how model performance varies with signal parameters (mediator mass and R_invisible values).

## Signal Processing

- The `sneaky5000R075` signal is automatically skipped due to known data length mismatches
- **Optional 5000 GeV filtering**: Use `--skip-5000` flag or `skip_5000=True` parameter to exclude all 5000 GeV signals from analysis
- Signal names are parsed to extract mediator mass and R_invisible values
- Each signal's performance metrics are collected across all models in the workspace

### Signal Parameter Encoding

The parameterized violin plots decode signal names as follows:
- `sneaky1000R025` → Mediator mass: 1000 GeV, R_invisible: 0.25
- `sneaky2000R05` → Mediator mass: 2000 GeV, R_invisible: 0.5
- `sneaky3000R075` → Mediator mass: 3000 GeV, R_invisible: 0.75
- etc.

This allows visualization of how model performance varies across:
- **Mediator masses**: 1000, 2000, 3000, 4000, 5000 GeV (or subset when using `--skip-5000`)
- **R_invisible values**: 0.25, 0.5, 0.75

### Color Encoding System

The enhanced parameterized plots use a sophisticated dual-encoding color system:
- **Primary colors by mass**: Each mass gets a unique, easily distinguishable color (red, yellow, green, blue, violet)
- **Transparency by R_invisible**: Higher R_invisible values become more transparent, creating a natural visual hierarchy
- **Consistent mapping**: Colors remain consistent across all plot types for easy comparison

## Dependencies

The statistical plotting functionality requires:
- `matplotlib` (for plotting)
- `numpy` (for numerical operations)
- Standard library modules: `os`, `re`, `warnings`, `collections`

Note: This functionality is kept in a separate module (`statistical_plotting.py`) to avoid adding dependencies to the main plotting module.

## File Structure

```
bead/src/utils/
├── plotting.py              # Main plotting functions (no new dependencies)
└── statistical_plotting.py  # Statistical plotting functions (requires additional deps)
```

## Error Handling

- Validates input file existence
- Handles missing data gracefully
- Provides informative error messages
- Includes optional verbose mode for debugging
- Skips problematic signals automatically

## Visualization Features

- **Right-aligned x-axis labels**: Model names are right-aligned to prevent overlap with long model names (e.g., "convvae_house_sc_anneal")
- **Enhanced color schemes**:
  - Basic plots use workspace-specific colors (CSF results) or matplotlib defaults
  - Parameterized plots use custom mass-based colors (red, yellow, green, blue, violet) with R_invisible transparency encoding
- **Professional layouts**: Proper spacing, legends, and grid lines for publication-ready plots
- **Parameter legends**: Parameterized plots include comprehensive legends for signal parameters showing both mass and R_invisible combinations
- **Multiple plot formats**: Six different plot types for comprehensive analysis:
  - Basic: Box, violin, combined
  - Parameterized: Box, violin, combined (with signal parameter color-coding)
- **Statistical robustness**: Proper handling of data grouping, outlier visualization, and distribution analysis
- **Performance insights**: Reveals both overall model performance and parameter-specific effects
- **Flexible data filtering**: Option to exclude 5000 GeV signals for focused analysis on lower mass ranges

## Future Enhancements

Potential improvements could include:
- Support for additional metrics beyond AUC and TPR
- Interactive plots with hover information
- Statistical significance testing between models
- Custom color scheme configuration via command-line arguments
- Export to other formats (PNG, SVG, etc.)
- Automated statistical reporting with confidence intervals
- Parameter correlation analysis
- Model ranking based on parameter-specific performance
- Advanced filtering options (e.g., exclude specific R_invisible values)
- Performance optimization for large datasets
