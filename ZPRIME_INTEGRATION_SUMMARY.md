# Zprime Signal Integration Summary

## Overview
Successfully implemented support for zprime signals in the BEAD statistical plotting system, allowing generation of per-signal ROC curves and statistical plots identical to the sneaky signal case.

## Changes Made

### 1. Signal Type Detection System
- **File**: `bead/src/utils/statistical_plotting.py`
- **Added**: `detect_signal_type()` function that automatically identifies:
  - `sneaky`: Signals matching `sneaky*` pattern
  - `zprime`: Signals matching `(h7|py8)zp\d+` pattern  
  - `unknown`: All other signals

### 2. Parameter Extraction Functions
- **Added**: `extract_sneaky_params()` - Extracts (mass, r_invisible) from sneaky signals
- **Added**: `extract_zprime_params()` - Extracts (mass, generator) from zprime signals
- **Added**: `get_signal_display_name()` - Returns display names ("Sneaky", "Z'", etc.)

### 3. Updated Signal Naming Convention
- **Zprime signals**: `h7zp1000`, `py8zp2000`, etc.
  - `h7` = Herwig generator
  - `py8` = Pythia generator
  - `zp` = Zprime
  - `1000` = Mass in GeV
- **Backward compatible**: All existing sneaky signal functionality preserved

### 4. Parameter Mapping
| Signal Type | Parameter 1 | Parameter 2 | Visual Encoding |
|-------------|-------------|-------------|-----------------|
| Sneaky      | Mass        | R_invisible | Color = Mass, Transparency = R_inv |
| Zprime      | Mass        | Generator   | Color = Mass, Transparency = Generator |

- **Sneaky R_invisible**: 0.25 (darker) → 0.75 (lighter)
- **Zprime Generator**: herwig (darker) → pythia (lighter)

### 5. Enhanced skip_5000 Functionality
- **Parsing level**: Skips known problematic signals (`sneaky5000R075`, `h7zp5000`, `py8zp5000`)
- **Plotting level**: `skip_5000=True` excludes any signal containing "5000"
- **Coverage**: Works for both sneaky and zprime signal types

### 6. Updated Functions
All four `extract_signal_params()` functions in statistical plotting functions updated:
- `create_parameterized_violin_plots()`
- `create_parameterized_box_plots()`  
- `create_parameterized_combined_plots()`
- Main plotting parameter handling

### 7. Display and Labeling
- **Plot titles**: Include signal type (e.g., "Model Performance - workspace (Z')")
- **Legend labels**: 
  - Sneaky: "1000GeV, R=0.25"
  - Zprime: "1000GeV, herwig"
- **Display names**: "Z'" for zprime signals in plots

## File Changes Summary

### Modified Files:
1. **`bead/src/utils/statistical_plotting.py`**
   - Added signal detection and parameter extraction functions
   - Updated all plotting functions to handle both signal types
   - Enhanced skip_5000 functionality
   - Updated color schemes and legends

### Test Files Created:
1. **`test_zprime_integration.py`** - Basic functionality tests
2. **`test_zprime_comprehensive.py`** - End-to-end integration tests
3. **`test_skip5000_direct.py`** - Skip_5000 functionality tests

## Usage Examples

### 1. Generating Per-Signal ROC Plots for Zprime

```bash
# Place your zprime CSV files in the workspace data directory
# Files should be named: sig_test_h7zp1000.csv, sig_test_py8zp2000.csv, etc.

# Run BEAD with roc_per_signal option to generate individual ROC plots
uv run bead -m plot -p your_workspace project_name -o roc_per_signal -v
```

### 2. Generating Statistical Plots from ROC Output

#### Command Line Usage:
```bash
# Generate all statistical plots from ROC output
uv run python paper_plots/generate_statistical_plots.py roc_output.txt ./zprime_plots/

# Generate plots excluding 5000 GeV signals (recommended if 5000 fails)
uv run python paper_plots/generate_statistical_plots.py roc_output.txt ./zprime_plots/ --skip-5000 -v
```

#### Python API Usage:
```python
from bead.src.utils.statistical_plotting import generate_statistical_plots_from_roc_output

# Basic usage
generate_statistical_plots_from_roc_output(
    "zprime_roc_output.txt", 
    save_dir="./zprime_plots/",
    verbose=True
)

# With 5000 GeV filtering
generate_statistical_plots_from_roc_output(
    "zprime_roc_output.txt", 
    save_dir="./zprime_plots/",
    skip_5000=True,  # Excludes h7zp5000, py8zp5000, etc.
    verbose=True
)
```

### 3. Example Workflow Script

```python
#!/usr/bin/env python3
"""
Complete zprime analysis workflow example.
"""

import os
import subprocess
from bead.src.utils.statistical_plotting import (
    generate_statistical_plots_from_roc_output,
    parse_roc_output
)

def run_zprime_analysis(workspace_name, project_name, zprime_data_dir):
    """Complete zprime analysis workflow."""
    
    print(f"🚀 Starting zprime analysis for {workspace_name}/{project_name}")
    
    # Step 1: Ensure data is in the right place
    data_dir = f"bead/workspaces/{workspace_name}/data/csv/"
    print(f"📁 Expected zprime data in: {data_dir}")
    print("   Required files: sig_test_h7zp1000.csv, sig_test_py8zp1000.csv, etc.")
    
    # Step 2: Run BEAD training and inference (if needed)
    print("🏋️  Running BEAD training and inference...")
    subprocess.run([
        "uv", "run", "bead", 
        "-m", "chain", 
        "-p", workspace_name, project_name,
        "-o", "convertcsv_prepareinputs_train_detect"
    ], check=True)
    
    # Step 3: Generate per-signal ROC plots
    print("📊 Generating per-signal ROC plots...")
    result = subprocess.run([
        "uv", "run", "bead",
        "-m", "plot",
        "-p", workspace_name, project_name,
        "-o", "roc_per_signal", "-v"
    ], capture_output=True, text=True)
    
    # Save ROC output for statistical plotting
    roc_output_file = f"{workspace_name}_{project_name}_roc_output.txt"
    with open(roc_output_file, 'w') as f:
        f.write(result.stdout)
    
    print(f"💾 ROC output saved to: {roc_output_file}")
    
    # Step 4: Generate statistical plots
    print("📈 Generating statistical plots...")
    plot_dir = f"zprime_plots_{workspace_name}_{project_name}"
    
    try:
        # Try without skip_5000 first
        generate_statistical_plots_from_roc_output(
            roc_output_file,
            save_dir=plot_dir,
            skip_5000=False,
            verbose=True
        )
        print(f"✅ Statistical plots saved to: {plot_dir}")
        
    except Exception as e:
        print(f"⚠️  Error with 5000 GeV signals: {e}")
        print("🔄 Retrying with skip_5000=True...")
        
        # Retry with skip_5000 if there are issues
        plot_dir_no5000 = f"{plot_dir}_no5000"
        generate_statistical_plots_from_roc_output(
            roc_output_file,
            save_dir=plot_dir_no5000,
            skip_5000=True,
            verbose=True
        )
        print(f"✅ Statistical plots (no 5000) saved to: {plot_dir_no5000}")
    
    print("🎉 Zprime analysis complete!")

# Example usage
if __name__ == "__main__":
    run_zprime_analysis(
        workspace_name="zprime_analysis",
        project_name="convvae_zprime",
        zprime_data_dir="/path/to/zprime/csvs"
    )
```

### 4. Expected ROC Output Format for Zprime

```text
Generating per-signal ROC plots...
Found 6 signal files for per-signal ROC plotting

Processing signal file: h7zp1000 (indices 1407905:1419763)
  h7zp1000 - LOSS AUC: 0.6364
    TPR at FPR 1.0e-04: 0.1234
    TPR at FPR 1.0e-03: 0.2345
    TPR at FPR 1.0e-02: 0.3456

Saved per-signal ROC plot: bead/workspaces/zprime_test/convvae/output/plots/loss/roc_h7zp1000.pdf

Processing signal file: py8zp1000 (indices 1419763:1428713)
  py8zp1000 - LOSS AUC: 0.6123
    TPR at FPR 1.0e-04: 0.1123
    TPR at FPR 1.0e-03: 0.2234
    TPR at FPR 1.0e-02: 0.3345

Saved per-signal ROC plot: bead/workspaces/zprime_test/convvae/output/plots/loss/roc_py8zp1000.pdf
```

### 5. File Structure for Zprime Data

```
bead/workspaces/your_workspace/
├── data/
│   └── csv/
│       ├── bkg_test_herwig.csv
│       ├── bkg_test_pythia.csv
│       ├── bkg_train_herwig.csv
│       ├── bkg_train_pythia.csv
│       ├── sig_test_h7zp1000.csv    # Herwig Z' 1000 GeV
│       ├── sig_test_py8zp1000.csv   # Pythia Z' 1000 GeV
│       ├── sig_test_h7zp2000.csv    # Herwig Z' 2000 GeV
│       ├── sig_test_py8zp2000.csv   # Pythia Z' 2000 GeV
│       ├── sig_test_h7zp3000.csv    # Herwig Z' 3000 GeV
│       ├── sig_test_py8zp3000.csv   # Pythia Z' 3000 GeV
│       ├── sig_test_h7zp4000.csv    # Herwig Z' 4000 GeV
│       ├── sig_test_py8zp4000.csv   # Pythia Z' 4000 GeV
│       ├── sig_test_h7zp5000.csv    # Herwig Z' 5000 GeV (may be skipped)
│       └── sig_test_py8zp5000.csv   # Pythia Z' 5000 GeV (may be skipped)
└── your_project/
    └── output/
        └── plots/
            └── loss/
                ├── roc_h7zp1000.pdf
                ├── roc_py8zp1000.pdf
                └── ... (individual ROC plots)
```

### 6. Batch Processing Multiple Models

```bash
#!/bin/bash
# batch_zprime_analysis.sh

WORKSPACE="zprime_study"
MODELS=("convvae" "convvae_planar" "convvae_house" "ntx_convvae")

echo "🚀 Starting batch zprime analysis..."

# Generate ROC output for each model
for model in "${MODELS[@]}"; do
    echo "📊 Processing model: $model"
    
    # Run per-signal ROC generation
    uv run bead -m plot -p $WORKSPACE $model -o roc_per_signal -v > "${WORKSPACE}_${model}_roc_output.txt"
    
    echo "✅ ROC output saved for $model"
done

# Generate combined statistical plots
echo "📈 Generating statistical plots..."
uv run python paper_plots/generate_statistical_plots.py \
    "${WORKSPACE}_*_roc_output.txt" \
    "./zprime_statistical_plots/" \
    --skip-5000 -v

echo "🎉 Batch analysis complete!"
```

## Statistical Plotting

### ROC Output Format
```
Processing signal file: h7zp1000 (indices 1407905:1419763)
  h7zp1000 - LOSS AUC: 0.6364
    TPR at FPR 1.0e-04: 0.1234
    ...
Saved per-signal ROC plot: bead/workspaces/zprime_test/convvae/output/plots/loss/roc_h7zp1000.pdf
```

### Statistical Plotting
```python
# Same API as before, auto-detects signal type
from bead.src.utils.statistical_plotting import generate_statistical_plots_from_roc_output

generate_statistical_plots_from_roc_output(
    "zprime_roc_output.txt", 
    save_dir="./zprime_plots/",
    skip_5000=True,  # Excludes any 5000 GeV signals
    verbose=True
)
```

## Verification Status
✅ **All tests pass**
- Signal type auto-detection works correctly
- Parameter extraction handles both signal types
- Statistical plotting generates correct visualizations
- Skip_5000 functionality works for both signal types
- Backward compatibility maintained for sneaky signals

## Integration Benefits
1. **Automatic**: No manual configuration required
2. **Backward Compatible**: Existing sneaky workflows unchanged
3. **Extensible**: Easy to add new signal types in the future
4. **Consistent**: Same visual encoding principles across signal types
5. **Robust**: Handles edge cases and problematic signals appropriately

## Ready for Production
The zprime integration is complete and ready for use with real zprime data files following the naming convention `sig_test_h7zp1000.csv`, `sig_test_py8zp2000.csv`, etc.