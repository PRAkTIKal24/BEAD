#!/usr/bin/env python3
"""
Statistical Analysis Script for BEAD ROC Results

This script generates box plots, violin plots, and combined visualizations 
from ROC output data to analyze statistical variation of different models 
across signal samples.

Usage:
    python generate_statistical_plots.py <roc_output_file> [output_directory]

Example:
    python generate_statistical_plots.py roc_output.txt ./plots/
"""

import argparse
import os
import sys

# Add the bead src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bead', 'src'))

from utils.statistical_plotting import generate_statistical_plots_from_roc_output


def main():
    parser = argparse.ArgumentParser(
        description='Generate statistical plots from ROC output data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script parses ROC output from the roc_per_signal mode and creates:
- Box plots showing statistical variation across models and signals
- Violin plots showing performance distributions
- Combined plots with both box and violin overlays

The plots are organized by workspace and use consistent color schemes.
For csf_results workspace, specific model colors are applied as requested.
        """
    )
    
    parser.add_argument(
        'roc_output_file',
        help='Path to the ROC output text file'
    )
    
    parser.add_argument(
        'output_directory',
        nargs='?',
        default=None,
        help='Directory to save plots (default: same directory as input file)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--skip-5000',
        action='store_true',
        help='Skip 5000 GeV signals in the analysis'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.roc_output_file):
        print(f"Error: ROC output file '{args.roc_output_file}' not found.")
        sys.exit(1)
    
    # Set output directory
    if args.output_directory is None:
        output_dir = os.path.dirname(os.path.abspath(args.roc_output_file))
        output_dir = os.path.join(output_dir, 'statistical_plots')
    else:
        output_dir = args.output_directory
    
    print(f"Generating statistical plots from: {args.roc_output_file}")
    print(f"Saving plots to: {output_dir}")
    
    try:
        generate_statistical_plots_from_roc_output(
            output_file_path=args.roc_output_file,
            save_dir=output_dir,
            verbose=args.verbose,
            skip_5000=args.skip_5000
        )
        
        print("\\nStatistical plot generation completed successfully!")
        
        # List generated files
        if os.path.exists(output_dir):
            pdf_files = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]
            if pdf_files:
                print(f"\\nGenerated {len(pdf_files)} plot files:")
                for file in sorted(pdf_files):
                    print(f"  - {file}")
            else:
                print("\\nWarning: No PDF files found in output directory.")
        
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()