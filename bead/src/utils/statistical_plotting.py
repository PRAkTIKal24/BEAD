"""
Statistical visualization utilities for model performance analysis.

This module provides functions for creating box plots, violin plots, and combined
visualizations to analyze statistical variation of different models across signal samples.
These plots are designed to work with ROC output data from the roc_per_signal mode.

Functions:
    parse_roc_output: Parse ROC output file to extract performance metrics.
    create_box_plots: Generate box plots showing statistical variation.
    create_violin_plots: Generate violin plots showing performance distributions.
    create_combined_box_violin_plots: Create overlaid box and violin plots.
    generate_statistical_plots_from_roc_output: Main function to generate all plots.

Note:
    This module requires pandas for data manipulation. It's kept separate from the
    main plotting module to avoid adding pandas as a core dependency.
"""

import os
import re
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def detect_signal_type(signal_name):
    """
    Detect the signal type based on signal name pattern.
    
    Parameters
    ----------
    signal_name : str
        The signal name to analyze
        
    Returns
    -------
    str
        Signal type: 'sneaky', 'zprime', or 'unknown'
    """
    if signal_name.startswith('sneaky'):
        return 'sneaky'
    elif re.match(r'(h7|py8)zp\d+', signal_name):
        return 'zprime'
    return 'unknown'


def extract_sneaky_params(signal_name):
    """
    Extract parameters from sneaky signal names.
    
    Parameters
    ----------
    signal_name : str
        Signal name like 'sneaky1000R025'
        
    Returns
    -------
    tuple
        (mass, r_inv) or (None, None) if parsing fails
    """
    match = re.match(r'sneaky(\d+)R(\d+)', signal_name)
    if match:
        mass = int(match.group(1))
        r_inv_str = match.group(2)
        # Convert R_invisible string to float (025 -> 0.25, 05 -> 0.5, 075 -> 0.75)
        if r_inv_str == '025':
            r_inv = 0.25
        elif r_inv_str == '05':
            r_inv = 0.5
        elif r_inv_str == '075':
            r_inv = 0.75
        else:
            r_inv = float(r_inv_str) / 100  # fallback
        return mass, r_inv
    return None, None


def extract_zprime_params(signal_name):
    """
    Extract parameters from zprime signal names.
    
    Parameters
    ----------
    signal_name : str
        Signal name like 'h7zp1000' or 'py8zp1000'
        
    Returns
    -------
    tuple
        (mass, generator) or (None, None) if parsing fails
    """
    match = re.match(r'(h7|py8)zp(\d+)', signal_name)
    if match:
        generator_code = match.group(1)
        mass = int(match.group(2))
        # Convert generator code to readable name
        generator = 'herwig' if generator_code == 'h7' else 'pythia'
        return mass, generator
    return None, None


def get_signal_display_name(signal_type):
    """
    Get display name for signal type.
    
    Parameters
    ----------
    signal_type : str
        Signal type ('sneaky', 'zprime', etc.)
        
    Returns
    -------
    str
        Display name for plots
    """
    if signal_type == 'sneaky':
        return 'Sneaky'
    elif signal_type == 'zprime':
        return "Z'"
    return signal_type.capitalize()


def parse_roc_output(output_file_path, verbose=False):
    """
    Parse ROC output file to extract AUC and TPR values for each model and signal.
    
    Parameters
    ----------
    output_file_path : str
        Path to the ROC output text file
    verbose : bool, optional
        Whether to print parsing progress, default is False
        
    Returns
    -------
    dict
        Dictionary with workspace names as keys, each containing model data
        Structure: {workspace_name: {model_name: [{signal_name: str, auc: float, 
                   tpr_1e-4: float, tpr_1e-3: float, tpr_1e-2: float}]}}
    """
    if verbose:
        print(f"Parsing ROC output from: {output_file_path}")
    
    data = defaultdict(lambda: defaultdict(list))
    
    with open(output_file_path, 'r') as file:
        content = file.read()
    
    # Split by sections that start with "Generating per-signal ROC plots..."
    sections = re.split(r'Generating per-signal ROC plots\.\.\.', content)
    
    for section in sections[1:]:  # Skip the first empty section
        current_workspace = None
        current_model = None
        
        lines = section.strip().split('\n')
        
        for i, line in enumerate(lines):
            # Look for signal processing lines
            signal_match = re.match(r'Processing signal file: (\w+) \(indices', line)
            if signal_match:
                signal_name = signal_match.group(1)
                
                # Skip known problematic 5000 GeV signals
                if (signal_name == "sneaky5000R075" or 
                    signal_name in ["h7zp5000", "py8zp5000"]):
                    continue
                
                # Look for the next lines containing AUC and TPR values
                if i + 1 < len(lines):
                    auc_line = lines[i + 1].strip()
                    auc_match = re.search(r'(\w+) - LOSS AUC: ([0-9.]+)', auc_line)
                    if auc_match:
                        auc_value = float(auc_match.group(2))
                        
                        # Extract TPR values from the next 3 lines  
                        tpr_values = {}
                        fpr_levels = ['1.0e-04', '1.0e-03', '1.0e-02']
                        
                        for j, fpr_level in enumerate(fpr_levels):
                            if i + 2 + j < len(lines):
                                tpr_line = lines[i + 2 + j].strip()
                                tpr_match = re.search(f'TPR at FPR {re.escape(fpr_level)}: ([0-9.]+)', tpr_line)
                                if tpr_match:
                                    tpr_values[f'tpr_{fpr_level}'] = float(tpr_match.group(1))
                        
                        # Look for the saved plot line to extract workspace and model
                        for k in range(i + 5, min(i + 8, len(lines))):
                            if k < len(lines) and 'Saved per-signal ROC plot:' in lines[k]:
                                plot_path = lines[k].split(': ')[1]
                                # Extract workspace and model from path like:
                                # bead/workspaces/csf_results/convvae/output/plots/loss/roc_sneaky1000R025.pdf
                                path_parts = plot_path.split('/')
                                if len(path_parts) >= 5 and path_parts[1] == 'workspaces':
                                    current_workspace = path_parts[2]
                                    current_model = path_parts[3]
                                    
                                    # Create the data entry
                                    entry = {
                                        'signal_name': signal_name,
                                        'auc': auc_value,
                                        **tpr_values
                                    }
                                    
                                    data[current_workspace][current_model].append(entry)
                                    
                                    if verbose:
                                        print(f"Added data for {current_workspace}/{current_model}/{signal_name}: AUC={auc_value:.4f}")
                                break
    
    return dict(data)


def _convert_to_dataframe(parsed_data, workspace_name, skip_5000=False):
    """
    Convert parsed data to a structured format for plotting.
    
    Parameters
    ----------
    parsed_data : dict
        Parsed data from parse_roc_output
    workspace_name : str
        Name of the workspace to convert
    skip_5000 : bool, optional
        Whether to skip 5000 GeV signals, default is False
    
    Returns
    -------
    list
        List of dictionaries containing model, signal, and metric data
    """
    workspace_data = parsed_data.get(workspace_name, {})
    plot_data = []
    
    for model_name, model_data in workspace_data.items():
        for entry in model_data:
            # Skip 5000 GeV signals if requested (applies to both sneaky and zprime)
            if skip_5000:
                signal_name = entry['signal_name']
                # Skip any signal containing "5000" (covers sneaky5000*, h7zp5000, py8zp5000, etc.)
                if '5000' in signal_name:
                    continue
                
            plot_data.append({
                'model': model_name,
                'signal': entry['signal_name'],
                'AUC': entry['auc'],
                'TPR_1e-4': entry.get('tpr_1.0e-04', np.nan),
                'TPR_1e-3': entry.get('tpr_1.0e-03', np.nan),
                'TPR_1e-2': entry.get('tpr_1.0e-02', np.nan)
            })
    
    return plot_data


def _sort_models_by_sc(models, workspace_name):
    """
    Sort models to group by tag (hp, ps, hs) with non-sc version before sc version for 2class workspace.
    
    Parameters
    ----------
    models : list
        List of model names
    workspace_name : str
        Name of the workspace
        
    Returns
    -------
    list
        Sorted list of models
    """
    if workspace_name == '2class':
        # Extract tag and sc status for each model
        model_info = []
        for model in models:
            # Look for hp, ps, or hs tag
            tag = None
            for prefix in ['hp', 'ps', 'hs']:
                if prefix in model:
                    tag = prefix
                    break
            
            # Check if model has sc
            has_sc = 'sc' in model
            
            model_info.append((tag, has_sc, model))
        
        # Sort by: tag (hp, ps, hs), then non-sc before sc, then alphabetically
        model_info.sort(key=lambda x: (x[0] or 'zzz', x[1], x[2]))
        
        return [model for _, _, model in model_info]
    else:
        # For other workspaces, just sort alphabetically
        return sorted(models)


def create_box_plots(parsed_data, save_dir, verbose=False, skip_5000=False):
    """
    Create box plots showing statistical variation across models and signals.
    
    Parameters
    ----------
    parsed_data : dict
        Parsed ROC data from parse_roc_output
    save_dir : str
        Directory to save the plots
    verbose : bool, optional
        Whether to print progress, default is False
    skip_5000 : bool, optional
        Whether to skip 5000 GeV signals, default is False
    """
    if verbose:
        print("Creating box plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Color scheme for csf_results models
    csf_colors = {
        'convvae': 'blue',
        'convvae_planar': 'orange', 
        'convvae_house': 'green',
        'ntx_convvae': 'red',
        'dvae': 'violet',
        'convvae_sc': 'brown',
        'convvae_house_sc_anneal': 'pink'
    }
    
    for workspace_name in parsed_data.keys():
        if verbose:
            print(f"Processing workspace: {workspace_name}")
        
        # Convert data to structured format
        plot_data = _convert_to_dataframe(parsed_data, workspace_name)
        
        if not plot_data:
            continue
        
        # Extract unique models and metrics
        models = list(set(entry['model'] for entry in plot_data))
        models = _sort_models_by_sc(models, workspace_name)  # Sort models by tag grouping
        metrics = ['AUC', 'TPR_1e-4', 'TPR_1e-3', 'TPR_1e-2']
        
        # Create figure with subplots for each metric
        fig, axes = plt.subplots(1, 4, figsize=(16, 6))
        if len(metrics) == 1:
            axes = [axes]
        
        # Determine colors
        if workspace_name == 'csf_results':
            colors = [csf_colors.get(model, 'gray') for model in models]
        else:
            # Use default matplotlib colors for other workspaces
            colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for box plot
            box_data = []
            box_labels = []
            box_colors = []
            
            for j, model in enumerate(models):
                model_values = [entry[metric] for entry in plot_data 
                              if entry['model'] == model and not np.isnan(entry[metric])]
                if len(model_values) > 0:
                    box_data.append(model_values)
                    box_labels.append(model)
                    box_colors.append(colors[j])
            
            if box_data:
                bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                               showfliers=True, flierprops={'marker': 'o', 'markersize': 4})
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax.set_title(f'{metric.replace("_", " at FPR ")}')
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            # Right-align x-axis labels to avoid overlap
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment('right')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Model Performance Comparison - {workspace_name}', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(save_dir, f'{workspace_name}_box_plots.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"Saved box plot: {save_path}")


def create_violin_plots(parsed_data, save_dir, verbose=False, skip_5000=False):
    """
    Create violin plots showing statistical variation across models and signals.
    
    Parameters
    ----------
    parsed_data : dict
        Parsed ROC data from parse_roc_output
    save_dir : str
        Directory to save the plots
    verbose : bool, optional
        Whether to print progress, default is False
    """
    if verbose:
        print("Creating violin plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Color scheme for csf_results models
    csf_colors = {
        'convvae': 'blue',
        'convvae_planar': 'orange', 
        'convvae_house': 'green',
        'ntx_convvae': 'red',
        'dvae': 'violet',
        'convvae_sc': 'brown',
        'convvae_house_sc_anneal': 'pink'
    }
    
    for workspace_name in parsed_data.keys():
        if verbose:
            print(f"Processing workspace: {workspace_name}")
        
        # Convert data to structured format
        plot_data = _convert_to_dataframe(parsed_data, workspace_name)
        
        if not plot_data:
            continue
        
        # Extract unique models and metrics
        models = list(set(entry['model'] for entry in plot_data))
        models = _sort_models_by_sc(models, workspace_name)  # Sort models by tag grouping
        metrics = ['AUC', 'TPR_1e-4', 'TPR_1e-3', 'TPR_1e-2']
        
        # Create figure with subplots for each metric
        fig, axes = plt.subplots(1, 4, figsize=(16, 6))
        if len(metrics) == 1:
            axes = [axes]
        
        # Determine colors
        if workspace_name == 'csf_results':
            colors = [csf_colors.get(model, 'gray') for model in models]
        else:
            # Use default matplotlib colors for other workspaces
            colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for violin plot
            violin_data = []
            violin_labels = []
            violin_colors = []
            
            for j, model in enumerate(models):
                model_values = [entry[metric] for entry in plot_data 
                              if entry['model'] == model and not np.isnan(entry[metric])]
                if len(model_values) > 0:
                    violin_data.append(model_values)
                    violin_labels.append(model)
                    violin_colors.append(colors[j])
            
            if violin_data:
                vp = ax.violinplot(violin_data, positions=range(1, len(violin_data) + 1), 
                                  showmeans=True, showmedians=True)
                
                # Color the violins
                for pc, color in zip(vp['bodies'], violin_colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                
                ax.set_xticks(range(1, len(violin_labels) + 1))
                ax.set_xticklabels(violin_labels, rotation=45, ha='right')
            
            ax.set_title(f'{metric.replace("_", " at FPR ")}')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Model Performance Distribution - {workspace_name}', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(save_dir, f'{workspace_name}_violin_plots.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"Saved violin plot: {save_path}")


def create_combined_box_violin_plots(parsed_data, save_dir, verbose=False, skip_5000=False):
    """
    Create combined plots with both box and violin plots overlaid.
    
    Parameters
    ----------
    parsed_data : dict
        Parsed ROC data from parse_roc_output
    save_dir : str
        Directory to save the plots
    verbose : bool, optional
        Whether to print progress, default is False
    """
    if verbose:
        print("Creating combined box and violin plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Color scheme for csf_results models
    csf_colors = {
        'convvae': 'blue',
        'convvae_planar': 'orange', 
        'convvae_house': 'green',
        'ntx_convvae': 'red',
        'dvae': 'violet',
        'convvae_sc': 'brown',
        'convvae_house_sc_anneal': 'pink'
    }
    
    for workspace_name in parsed_data.keys():
        if verbose:
            print(f"Processing workspace: {workspace_name}")
        
        # Convert data to structured format
        plot_data = _convert_to_dataframe(parsed_data, workspace_name)
        
        if not plot_data:
            continue
        
        # Extract unique models and metrics
        models = list(set(entry['model'] for entry in plot_data))
        models = _sort_models_by_sc(models, workspace_name)  # Sort models by tag grouping
        metrics = ['AUC', 'TPR_1e-4', 'TPR_1e-3', 'TPR_1e-2']
        
        # Create figure with subplots for each metric
        fig, axes = plt.subplots(1, 4, figsize=(16, 6))
        if len(metrics) == 1:
            axes = [axes]
        
        # Determine colors
        if workspace_name == 'csf_results':
            colors = [csf_colors.get(model, 'gray') for model in models]
        else:
            # Use default matplotlib colors for other workspaces
            colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for both plots
            plot_data_list = []
            plot_labels = []
            plot_colors = []
            
            for j, model in enumerate(models):
                model_values = [entry[metric] for entry in plot_data 
                              if entry['model'] == model and not np.isnan(entry[metric])]
                if len(model_values) > 0:
                    plot_data_list.append(model_values)
                    plot_labels.append(model)
                    plot_colors.append(colors[j])
            
            if plot_data_list:
                positions = range(1, len(plot_data_list) + 1)
                
                # Create violin plot first (background)
                vp = ax.violinplot(plot_data_list, positions=positions, 
                                  showmeans=False, showmedians=False, showextrema=False)
                
                # Color the violins with transparency
                for pc, color in zip(vp['bodies'], plot_colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.4)
                
                # Create box plot on top
                bp = ax.boxplot(plot_data_list, positions=positions, patch_artist=True, 
                               showfliers=True, flierprops={'marker': 'o', 'markersize': 3},
                               widths=0.3)  # Make boxes narrower
                
                # Color the boxes with higher opacity
                for patch, color in zip(bp['boxes'], plot_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.8)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1)
                
                ax.set_xticks(positions)
                ax.set_xticklabels(plot_labels, rotation=45, ha='right')
            
            ax.set_title(f'{metric.replace("_", " at FPR ")}')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Model Performance (Box + Violin) - {workspace_name}', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(save_dir, f'{workspace_name}_combined_plots.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"Saved combined plot: {save_path}")


def create_parameterized_violin_plots(parsed_data, save_dir, verbose=False, skip_5000=False):
    """
    Create violin plots showing performance distributions subdivided by signal parameters.
    
    This function creates violin plots where each model's violin is subdivided to show
    how performance varies across different mediator masses and R_invisible values.
    Each violin shows both the overall distribution and parameter-specific patterns.
    
    Parameters
    ----------
    parsed_data : dict
        Parsed ROC data from parse_roc_output
    save_dir : str
        Directory to save the plots
    verbose : bool, optional
        Whether to print progress, default is False
    """
    if verbose:
        print("Creating parameterized violin plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Color scheme for csf_results models
    csf_colors = {
        'convvae': 'blue',
        'convvae_planar': 'orange', 
        'convvae_house': 'green',
        'ntx_convvae': 'red',
        'dvae': 'violet',
        'convvae_sc': 'brown',
        'convvae_house_sc_anneal': 'pink'
    }
    
    # Parameter extraction function
    def extract_signal_params(signal_name):
        """Extract parameters from signal name based on signal type."""
        signal_type = detect_signal_type(signal_name)
        
        if signal_type == 'sneaky':
            return extract_sneaky_params(signal_name)
        elif signal_type == 'zprime':
            return extract_zprime_params(signal_name)
        
        return None, None
    
    for workspace_name in parsed_data.keys():
        if verbose:
            print(f"Processing workspace: {workspace_name}")
        
        # Convert data to structured format with parameters
        plot_data = _convert_to_dataframe(parsed_data, workspace_name)
        
        if not plot_data:
            continue
        
        # Add parameter columns
        for entry in plot_data:
            mass, r_inv = extract_signal_params(entry['signal'])
            entry['mass'] = mass
            entry['r_inv'] = r_inv
        
        # Filter out entries with missing parameters
        plot_data = [entry for entry in plot_data if entry['mass'] is not None]
        
        # Apply skip_5000 filtering if requested
        if skip_5000:
            plot_data = [entry for entry in plot_data if entry["mass"] != 5000]
        
        if not plot_data:
            continue
        
        # Extract unique models and metrics
        models = list(set(entry['model'] for entry in plot_data))
        models = _sort_models_by_sc(models, workspace_name)  # Sort models by tag grouping
        metrics = ['AUC', 'TPR_1e-4', 'TPR_1e-3', 'TPR_1e-2']
        
        # Get unique parameter combinations for color coding
        masses = sorted(list(set(entry['mass'] for entry in plot_data if entry['mass'] is not None)))
        r_invs = sorted(list(set(entry['r_inv'] for entry in plot_data if entry['r_inv'] is not None)))
        
        # Create color scheme: different colors for masses, different brightness for R_inv
        # Custom color scheme for masses
        custom_mass_colors = {1000: "red", 2000: "yellow", 3000: "green", 4000: "blue", 5000: "violet"}
        # Convert color names to RGB tuples for proper alpha blending
        import matplotlib.colors as mcolors
        mass_color_map = {}
        for mass in masses:
            color_name = custom_mass_colors.get(mass, "gray")
            mass_color_map[mass] = mcolors.to_rgb(color_name)
        
        # R_inv brightness mapping (0.25 -> darkest, 0.75 -> lightest)
        r_inv_alpha_map = {0.25: 0.9, 0.5: 0.7, 0.75: 0.5}  # Higher alpha = more opaque = darker
        
        # Create figure with subplots for each metric
        fig, axes = plt.subplots(1, 4, figsize=(20, 8))
        if len(metrics) == 1:
            axes = [axes]
        
        # Determine main model colors
        if workspace_name == 'csf_results':
            model_colors = {model: csf_colors.get(model, 'gray') for model in models}
        else:
            model_color_list = plt.cm.tab10(np.linspace(0, 1, len(models)))
            model_colors = {model: color for model, color in zip(models, model_color_list)}
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for violin plot
            violin_data = []
            violin_labels = []
            violin_positions = []
            
            # Create separate violins for each model
            for j, model in enumerate(models):
                model_values = [entry[metric] for entry in plot_data 
                              if entry['model'] == model and not np.isnan(entry[metric])]
                if len(model_values) > 0:
                    violin_data.append(model_values)
                    violin_labels.append(model)
                    violin_positions.append(j + 1)
            
            if violin_data:
                # Create main violins
                vp = ax.violinplot(violin_data, positions=violin_positions, 
                                  showmeans=True, showmedians=True, widths=0.8)
                
                # Color the main violins with model colors (semi-transparent)
                for j, (pc, model) in enumerate(zip(vp['bodies'], violin_labels)):
                    pc.set_facecolor(model_colors[model])
                    pc.set_alpha(0.3)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(0.5)
                
                # Add parameter-specific scatter points with jitter
                for j, model in enumerate(violin_labels):
                    model_data = [entry for entry in plot_data 
                                if entry['model'] == model and not np.isnan(entry[metric])]
                    
                    # Group by parameters
                    param_groups = {}
                    for entry in model_data:
                        key = (entry['mass'], entry['r_inv'])
                        if key not in param_groups:
                            param_groups[key] = []
                        param_groups[key].append(entry[metric])
                    
                    # Plot each parameter group with jitter
                    for (mass, r_inv), values in param_groups.items():
                        # Add small horizontal jitter for visibility
                        jitter = np.random.normal(0, 0.02, len(values))
                        x_pos = [violin_positions[j] + jit for jit in jitter]
                        
                        # Get color and alpha based on mass and R_inv
                        base_color = mass_color_map[mass]
                        alpha = r_inv_alpha_map.get(r_inv, 0.7)
                        
                        ax.scatter(x_pos, values, 
                                 color=base_color, alpha=alpha,
                                 s=30, edgecolors='black', linewidth=0.3,
                                 label=f'{mass}GeV, R={r_inv}' if j == 0 else "")
                
                ax.set_xticks(violin_positions)
                ax.set_xticklabels(violin_labels, rotation=45, ha='right')
            
            ax.set_title(f'{metric.replace("_", " at FPR ")}')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        # Create legend for parameters (organized by mass with different transparencies for R_inv)
        handles = []
        labels = []
        for mass in masses:
            for r_inv in r_invs:
                if (mass, r_inv) in [(entry['mass'], entry['r_inv']) for entry in plot_data]:
                    base_color = mass_color_map[mass]
                    alpha = r_inv_alpha_map.get(r_inv, 0.7)
                    handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=base_color, markersize=8, alpha=alpha,
                                            markeredgecolor='black', markeredgewidth=0.3))
                    labels.append(f'{mass}GeV, R={r_inv}')
        
        # Add legend outside the plot area
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                  title='Signal Parameters', title_fontsize=12)
        
        plt.suptitle(f'Model Performance by Signal Parameters - {workspace_name}', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(save_dir, f'{workspace_name}_parameterized_violin_plots.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"Saved parameterized violin plot: {save_path}")


def create_parameterized_box_plots(parsed_data, save_dir, verbose=False, skip_5000=False):
    """
    Create box plots showing performance distributions subdivided by signal parameters.
    
    This function creates box plots where each model has separate boxes for different
    signal parameter combinations, with colors representing mediator mass and
    transparency representing R_invisible values.
    
    Parameters
    ----------
    parsed_data : dict
        Parsed ROC data from parse_roc_output
    save_dir : str
        Directory to save the plots
    verbose : bool, optional
        Whether to print progress, default is False
    """
    if verbose:
        print("Creating parameterized box plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Color scheme for csf_results models
    csf_colors = {
        'convvae': 'blue',
        'convvae_planar': 'orange', 
        'convvae_house': 'green',
        'ntx_convvae': 'red',
        'dvae': 'violet',
        'convvae_sc': 'brown',
        'convvae_house_sc_anneal': 'pink'
    }
    
    # Parameter extraction function
    def extract_signal_params(signal_name):
        """Extract parameters from signal name based on signal type."""
        signal_type = detect_signal_type(signal_name)
        
        if signal_type == 'sneaky':
            return extract_sneaky_params(signal_name)
        elif signal_type == 'zprime':
            return extract_zprime_params(signal_name)
        
        return None, None
    
    for workspace_name in parsed_data.keys():
        if verbose:
            print(f"Processing workspace: {workspace_name}")
        
        # Convert data to structured format with parameters
        plot_data = _convert_to_dataframe(parsed_data, workspace_name)
        
        if not plot_data:
            continue
        
        # Add parameter columns
        for entry in plot_data:
            mass, r_inv = extract_signal_params(entry['signal'])
            entry['mass'] = mass
            entry['r_inv'] = r_inv
        
        # Filter out entries with missing parameters
        plot_data = [entry for entry in plot_data if entry['mass'] is not None]
        
        # Apply skip_5000 filtering if requested
        if skip_5000:
            plot_data = [entry for entry in plot_data if entry["mass"] != 5000]
        
        if not plot_data:
            continue
        
        # Extract unique models and metrics
        models = list(set(entry['model'] for entry in plot_data))
        models = _sort_models_by_sc(models, workspace_name)  # Sort models by tag grouping
        metrics = ['AUC', 'TPR_1e-4', 'TPR_1e-3', 'TPR_1e-2']
        
        # Get unique parameter combinations for color coding
        masses = sorted(list(set(entry['mass'] for entry in plot_data if entry['mass'] is not None)))
        r_invs = sorted(list(set(entry['r_inv'] for entry in plot_data if entry['r_inv'] is not None)))
        
        # Create color scheme: different colors for masses, different brightness for R_inv
        # Custom color scheme for masses
        custom_mass_colors = {1000: "red", 2000: "yellow", 3000: "green", 4000: "blue", 5000: "violet"}
        # Convert color names to RGB tuples for proper alpha blending
        import matplotlib.colors as mcolors
        mass_color_map = {}
        for mass in masses:
            color_name = custom_mass_colors.get(mass, "gray")
            mass_color_map[mass] = mcolors.to_rgb(color_name)
        
        # R_inv brightness mapping (0.25 -> darkest, 0.75 -> lightest)
        r_inv_alpha_map = {0.25: 0.9, 0.5: 0.7, 0.75: 0.5}  # Higher alpha = more opaque = darker
        
        # Create figure with subplots for each metric
        fig, axes = plt.subplots(1, 4, figsize=(20, 8))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for box plot - separate boxes for each (model, mass, r_inv) combination
            box_data = []
            box_labels = []
            box_colors = []
            box_positions = []
            
            pos = 1
            for model in models:
                model_start_pos = pos
                for mass in masses:
                    for r_inv in r_invs:
                        # Get data for this specific combination
                        combo_values = [entry[metric] for entry in plot_data 
                                      if (entry['model'] == model and 
                                          entry['mass'] == mass and 
                                          entry['r_inv'] == r_inv and 
                                          not np.isnan(entry[metric]))]
                        
                        if len(combo_values) > 0:
                            box_data.append(combo_values)
                            box_labels.append(f'{model}\n{mass}GeV\nR={r_inv}')
                            
                            # Get color and alpha based on mass and R_inv
                            base_color = mass_color_map[mass]
                            alpha = r_inv_alpha_map.get(r_inv, 0.7)
                            box_colors.append((*base_color[:3], alpha))  # RGBA format
                            box_positions.append(pos)
                            pos += 1
                
                # Add some space between models
                if model != models[-1]:  # Don't add space after last model
                    pos += 0.5
            
            if box_data:
                bp = ax.boxplot(box_data, positions=box_positions, patch_artist=True, 
                               showfliers=True, flierprops={'marker': 'o', 'markersize': 3},
                               widths=0.6)
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(color)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(0.5)
                
                # Set custom x-axis labels with model names only (simplified)
                model_positions = []
                model_labels = []
                current_pos = 1
                for model in models:
                    # Find the center position for this model's boxes
                    model_box_positions = [pos for j, pos in enumerate(box_positions) 
                                         if box_labels[j].startswith(model)]
                    if model_box_positions:
                        center_pos = (min(model_box_positions) + max(model_box_positions)) / 2
                        model_positions.append(center_pos)
                        model_labels.append(model)
                
                ax.set_xticks(model_positions)
                ax.set_xticklabels(model_labels, rotation=45, ha='right')
            
            ax.set_title(f'{metric.replace("_", " at FPR ")}')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        # Create legend for parameters (organized by mass with different transparencies for R_inv)
        handles = []
        labels = []
        for mass in masses:
            for r_inv in r_invs:
                if any(entry['mass'] == mass and entry['r_inv'] == r_inv for entry in plot_data):
                    base_color = mass_color_map[mass]
                    alpha = r_inv_alpha_map.get(r_inv, 0.7)
                    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=(*base_color[:3], alpha),
                                               edgecolor='black', linewidth=0.5))
                    labels.append(f'{mass}GeV, R={r_inv}')
        
        # Add legend outside the plot area
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                  title='Signal Parameters', title_fontsize=12)
        
        plt.suptitle(f'Model Performance by Signal Parameters (Box) - {workspace_name}', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(save_dir, f'{workspace_name}_parameterized_box_plots.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"Saved parameterized box plot: {save_path}")


def create_parameterized_combined_plots(parsed_data, save_dir, verbose=False, skip_5000=False):
    """
    Create violin plots showing performance distributions subdivided by signal parameters.
    
    This function creates violin plots where each model's violin is subdivided to show
    how performance varies across different mediator masses and R_invisible values.
    Each violin shows both the overall distribution and parameter-specific patterns.
    
    Parameters
    ----------
    parsed_data : dict
        Parsed ROC data from parse_roc_output
    save_dir : str
        Directory to save the plots
    verbose : bool, optional
        Whether to print progress, default is False
    """
    if verbose:
        print("Creating parameterized violin plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Color scheme for csf_results models
    csf_colors = {
        'convvae': 'blue',
        'convvae_planar': 'orange', 
        'convvae_house': 'green',
        'ntx_convvae': 'red',
        'dvae': 'violet',
        'convvae_sc': 'brown',
        'convvae_house_sc_anneal': 'pink'
    }
    
    # Parameter extraction function
    def extract_signal_params(signal_name):
        """Extract parameters from signal name based on signal type."""
        signal_type = detect_signal_type(signal_name)
        
        if signal_type == 'sneaky':
            return extract_sneaky_params(signal_name)
        elif signal_type == 'zprime':
            return extract_zprime_params(signal_name)
        
        return None, None
    
    for workspace_name in parsed_data.keys():
        if verbose:
            print(f"Processing workspace: {workspace_name}")
        
        # Convert data to structured format with parameters
        plot_data = _convert_to_dataframe(parsed_data, workspace_name)
        
        if not plot_data:
            continue
        
        # Add parameter columns
        for entry in plot_data:
            mass, r_inv = extract_signal_params(entry['signal'])
            entry['mass'] = mass
            entry['r_inv'] = r_inv
        
        # Filter out entries with missing parameters
        plot_data = [entry for entry in plot_data if entry['mass'] is not None]
        
        # Apply skip_5000 filtering if requested
        if skip_5000:
            plot_data = [entry for entry in plot_data if entry["mass"] != 5000]
        
        if not plot_data:
            continue
        
        # Extract unique models and metrics
        models = list(set(entry['model'] for entry in plot_data))
        models = _sort_models_by_sc(models, workspace_name)  # Sort models by tag grouping
        metrics = ['AUC', 'TPR_1e-4', 'TPR_1e-3', 'TPR_1e-2']
        
        # Get unique parameter combinations for color coding
        masses = sorted(list(set(entry['mass'] for entry in plot_data if entry['mass'] is not None)))
        r_invs = sorted(list(set(entry['r_inv'] for entry in plot_data if entry['r_inv'] is not None)))
        
        # Create color scheme: different colors for masses, different brightness for R_inv
        # Custom color scheme for masses
        custom_mass_colors = {1000: "red", 2000: "yellow", 3000: "green", 4000: "blue", 5000: "violet"}
        # Convert color names to RGB tuples for proper alpha blending
        import matplotlib.colors as mcolors
        mass_color_map = {}
        for mass in masses:
            color_name = custom_mass_colors.get(mass, "gray")
            mass_color_map[mass] = mcolors.to_rgb(color_name)
        
        # R_inv brightness mapping (0.25 -> darkest, 0.75 -> lightest)
        r_inv_alpha_map = {0.25: 0.9, 0.5: 0.7, 0.75: 0.5}  # Higher alpha = more opaque = darker
        
        # Create figure with subplots for each metric
        fig, axes = plt.subplots(1, 4, figsize=(20, 8))
        if len(metrics) == 1:
            axes = [axes]
        
        # Determine main model colors
        if workspace_name == 'csf_results':
            model_colors = {model: csf_colors.get(model, 'gray') for model in models}
        else:
            model_color_list = plt.cm.tab10(np.linspace(0, 1, len(models)))
            model_colors = {model: color for model, color in zip(models, model_color_list)}
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for violin plot
            violin_data = []
            violin_labels = []
            violin_positions = []
            
            # Create separate violins for each model
            for j, model in enumerate(models):
                model_values = [entry[metric] for entry in plot_data 
                              if entry['model'] == model and not np.isnan(entry[metric])]
                if len(model_values) > 0:
                    violin_data.append(model_values)
                    violin_labels.append(model)
                    violin_positions.append(j + 1)
            
            if violin_data:
                # Create main violins
                vp = ax.violinplot(violin_data, positions=violin_positions, 
                                  showmeans=True, showmedians=True, widths=0.8)
                
                # Color the main violins with model colors (semi-transparent)
                for j, (pc, model) in enumerate(zip(vp['bodies'], violin_labels)):
                    pc.set_facecolor(model_colors[model])
                    pc.set_alpha(0.3)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(0.5)
                
                # Add parameter-specific scatter points with jitter
                for j, model in enumerate(violin_labels):
                    model_data = [entry for entry in plot_data 
                                if entry['model'] == model and not np.isnan(entry[metric])]
                    
                    # Group by parameters
                    param_groups = {}
                    for entry in model_data:
                        key = (entry['mass'], entry['r_inv'])
                        if key not in param_groups:
                            param_groups[key] = []
                        param_groups[key].append(entry[metric])
                    
                    # Plot each parameter group with jitter
                    for (mass, r_inv), values in param_groups.items():
                        # Add small horizontal jitter for visibility
                        jitter = np.random.normal(0, 0.02, len(values))
                        x_pos = [violin_positions[j] + jit for jit in jitter]
                        
                        # Get color and alpha based on mass and R_inv
                        base_color = mass_color_map[mass]
                        alpha = r_inv_alpha_map.get(r_inv, 0.7)
                        
                        ax.scatter(x_pos, values, 
                                 color=base_color, alpha=alpha,
                                 s=30, edgecolors='black', linewidth=0.3,
                                 label=f'{mass}GeV, R={r_inv}' if j == 0 else "")
                
                ax.set_xticks(violin_positions)
                ax.set_xticklabels(violin_labels, rotation=45, ha='right')
            
            ax.set_title(f'{metric.replace("_", " at FPR ")}')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        # Create legend for parameters (organized by mass with different transparencies for R_inv)
        handles = []
        labels = []
        for mass in masses:
            for r_inv in r_invs:
                if (mass, r_inv) in [(entry['mass'], entry['r_inv']) for entry in plot_data]:
                    base_color = mass_color_map[mass]
                    alpha = r_inv_alpha_map.get(r_inv, 0.7)
                    handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=base_color, markersize=8, alpha=alpha,
                                            markeredgecolor='black', markeredgewidth=0.3))
                    labels.append(f'{mass}GeV, R={r_inv}')
        
        # Add legend outside the plot area
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                  title='Signal Parameters', title_fontsize=12)
        
        plt.suptitle(f'Model Performance by Signal Parameters - {workspace_name}', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(save_dir, f'{workspace_name}_parameterized_violin_plots.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"Saved parameterized violin plot: {save_path}")


def create_parameterized_combined_plots(parsed_data, save_dir, verbose=False, skip_5000=False):
    """
    Create combined plots with both parameterized violin and box plots overlaid.
    
    This function creates combined visualizations showing both the overall distribution
    (violin) and parameter-specific patterns (colored scatter + mini boxes) for each model.
    
    Parameters
    ----------
    parsed_data : dict
        Parsed ROC data from parse_roc_output
    save_dir : str
        Directory to save the plots
    verbose : bool, optional
        Whether to print progress, default is False
    """
    if verbose:
        print("Creating parameterized combined plots...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Color scheme for csf_results models
    csf_colors = {
        'convvae': 'blue',
        'convvae_planar': 'orange', 
        'convvae_house': 'green',
        'ntx_convvae': 'red',
        'dvae': 'violet',
        'convvae_sc': 'brown',
        'convvae_house_sc_anneal': 'pink'
    }
    
    # Parameter extraction function
    def extract_signal_params(signal_name):
        """Extract parameters from signal name based on signal type."""
        signal_type = detect_signal_type(signal_name)
        
        if signal_type == 'sneaky':
            return extract_sneaky_params(signal_name)
        elif signal_type == 'zprime':
            return extract_zprime_params(signal_name)
        
        return None, None
    
    for workspace_name in parsed_data.keys():
        if verbose:
            print(f"Processing workspace: {workspace_name}")
        
        # Convert data to structured format with parameters
        plot_data = _convert_to_dataframe(parsed_data, workspace_name)
        
        if not plot_data:
            continue
        
        # Add parameter columns
        for entry in plot_data:
            mass, r_inv = extract_signal_params(entry['signal'])
            entry['mass'] = mass
            entry['r_inv'] = r_inv
        
        # Filter out entries with missing parameters
        plot_data = [entry for entry in plot_data if entry['mass'] is not None]
        
        # Apply skip_5000 filtering if requested
        if skip_5000:
            plot_data = [entry for entry in plot_data if entry["mass"] != 5000]
        
        if not plot_data:
            continue
        
        # Extract unique models and metrics
        models = list(set(entry['model'] for entry in plot_data))
        models = _sort_models_by_sc(models, workspace_name)  # Sort models by tag grouping
        metrics = ['AUC', 'TPR_1e-4', 'TPR_1e-3', 'TPR_1e-2']
        
        # Get unique parameter combinations for color coding
        masses = sorted(list(set(entry['mass'] for entry in plot_data if entry['mass'] is not None)))
        r_invs = sorted(list(set(entry['r_inv'] for entry in plot_data if entry['r_inv'] is not None)))
        
        # Create color scheme: different colors for masses, different brightness for R_inv
        # Create custom color scheme: specific colors for masses
        custom_mass_colors = {
            1000: 'red',     
            2000: 'yellow', 
            3000: 'green',
            4000: 'blue', 
            5000: 'violet'
        }
        # Convert color names to RGB tuples for proper alpha blending
        import matplotlib.colors as mcolors
        mass_color_map = {}
        for mass in masses:
            color_name = custom_mass_colors.get(mass, 'gray')
            mass_color_map[mass] = mcolors.to_rgb(color_name)
        
        # R_inv brightness mapping
        r_inv_alpha_map = {0.25: 0.9, 0.5: 0.7, 0.75: 0.5}
        
        # Create figure with subplots for each metric
        fig, axes = plt.subplots(1, 4, figsize=(20, 8))
        if len(metrics) == 1:
            axes = [axes]
        
        # Determine main model colors
        if workspace_name == 'csf_results':
            model_colors = {model: csf_colors.get(model, 'gray') for model in models}
        else:
            model_color_list = plt.cm.tab10(np.linspace(0, 1, len(models)))
            model_colors = {model: color for model, color in zip(models, model_color_list)}
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for violin plot (background)
            violin_data = []
            violin_labels = []
            violin_positions = []
            
            for j, model in enumerate(models):
                model_values = [entry[metric] for entry in plot_data 
                              if entry['model'] == model and not np.isnan(entry[metric])]
                if len(model_values) > 0:
                    violin_data.append(model_values)
                    violin_labels.append(model)
                    violin_positions.append(j + 1)
            
            if violin_data:
                # Create main violins (background)
                vp = ax.violinplot(violin_data, positions=violin_positions, 
                                  showmeans=False, showmedians=False, showextrema=False,
                                  widths=0.8)
                
                # Color the main violins with model colors (very transparent)
                for j, (pc, model) in enumerate(zip(vp['bodies'], violin_labels)):
                    pc.set_facecolor(model_colors[model])
                    pc.set_alpha(0.2)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(0.3)
                
                # Add parameter-specific scatter points and mini box plots
                for j, model in enumerate(violin_labels):
                    model_data = [entry for entry in plot_data 
                                if entry['model'] == model and not np.isnan(entry[metric])]
                    
                    # Group by parameters
                    param_groups = {}
                    for entry in model_data:
                        key = (entry['mass'], entry['r_inv'])
                        if key not in param_groups:
                            param_groups[key] = []
                        param_groups[key].append(entry[metric])
                    
                    # Plot each parameter group with scatter + mini boxes
                    box_offset = -0.3
                    for (mass, r_inv), values in param_groups.items():
                        # Get color and alpha based on mass and R_inv
                        base_color = mass_color_map[mass]
                        alpha = r_inv_alpha_map.get(r_inv, 0.7)
                        
                        # Add scatter points with jitter
                        jitter = np.random.normal(0, 0.02, len(values))
                        x_pos = [violin_positions[j] + jit for jit in jitter]
                        
                        ax.scatter(x_pos, values, 
                                 color=base_color, alpha=alpha,
                                 s=25, edgecolors='black', linewidth=0.2,
                                 label=f'{mass}GeV, R={r_inv}' if j == 0 else "")
                        
                        # Add mini box plot for this parameter combination
                        mini_box_pos = violin_positions[j] + box_offset
                        bp_mini = ax.boxplot([values], positions=[mini_box_pos], 
                                           patch_artist=True, widths=0.05,
                                           showfliers=False, showcaps=False)
                        
                        # Color the mini box
                        bp_mini['boxes'][0].set_facecolor((*base_color[:3], alpha))
                        bp_mini['boxes'][0].set_edgecolor('black')
                        bp_mini['boxes'][0].set_linewidth(0.3)
                        
                        box_offset += 0.12  # Offset for next parameter combination
                
                ax.set_xticks(violin_positions)
                ax.set_xticklabels(violin_labels, rotation=45, ha='right')
            
            ax.set_title(f'{metric.replace("_", " at FPR ")}')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
        
        # Create legend for parameters
        handles = []
        labels = []
        for mass in masses:
            for r_inv in r_invs:
                if any(entry['mass'] == mass and entry['r_inv'] == r_inv for entry in plot_data):
                    base_color = mass_color_map[mass]
                    alpha = r_inv_alpha_map.get(r_inv, 0.7)
                    handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=base_color, markersize=8, alpha=alpha,
                                            markeredgecolor='black', markeredgewidth=0.3))
                    labels.append(f'{mass}GeV, R={r_inv}')
        
        # Add legend outside the plot area
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                  title='Signal Parameters', title_fontsize=12)
        
        plt.suptitle(f'Model Performance by Signal Parameters (Combined) - {workspace_name}', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(save_dir, f'{workspace_name}_parameterized_combined_plots.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"Saved parameterized combined plot: {save_path}")


def generate_statistical_plots_from_roc_output(output_file_path, save_dir=None, verbose=False, skip_5000=False):
    """
    Generate all statistical plots (box, violin, and combined) from ROC output file.
    
    Parameters
    ----------
    output_file_path : str
        Path to the ROC output text file
    save_dir : str, optional
        Directory to save plots, if None uses current directory
    verbose : bool, optional
        Whether to print progress, default is False
    skip_5000 : bool, optional
        Whether to skip 5000 GeV signals, default is False
    """
    if save_dir is None:
        save_dir = os.path.dirname(output_file_path)
    
    if verbose:
        print("Generating statistical plots from ROC output...")
    
    # Parse the ROC output
    parsed_data = parse_roc_output(output_file_path, verbose=verbose)
    
    if not parsed_data:
        print("No data found in ROC output file")
        return
    
    # Create all types of plots
    create_box_plots(parsed_data, save_dir, verbose=verbose, skip_5000=skip_5000)
    create_violin_plots(parsed_data, save_dir, verbose=verbose, skip_5000=skip_5000)
    create_combined_box_violin_plots(parsed_data, save_dir, verbose=verbose, skip_5000=skip_5000)
    create_parameterized_violin_plots(parsed_data, save_dir, verbose=verbose, skip_5000=skip_5000)
    create_parameterized_box_plots(parsed_data, save_dir, verbose=verbose, skip_5000=skip_5000)
    create_parameterized_combined_plots(parsed_data, save_dir, verbose=verbose, skip_5000=skip_5000)
    
    if verbose:
        print(f"All statistical plots saved to: {save_dir}")