#!/usr/bin/env python3
"""
Enhanced script to add/update test loss histogram config in multiple config files.

This script now intelligently handles existing configurations:
- If config doesn't exist: Adds new config with default values
- If config exists with matching values: Does nothing
- If config exists with different values: Updates to match the snippet values

Usage:
1. Save this script in your BEAD repository root
2. Run: python add_histogram_to_configs.py /path/to/your/workspace/configs/
   OR just run it and it will search common locations automatically
"""

import glob
import os
import re
import shutil
from datetime import datetime


def find_config_files(search_paths=None):
    """Find all config files in specified paths or common locations."""
    if search_paths is None:
        # Common patterns where config files might be located
        search_paths = [
            "bead/workspaces/*/*/config/*.py",
            "workspaces/*/*/config/*.py",
            "*/config/*.py",
            "*/*/*/config/*.py",
        ]

    config_files = []
    for pattern in search_paths:
        found = glob.glob(pattern, recursive=True)
        config_files.extend([f for f in found if os.path.isfile(f)])

    return list(set(config_files))  # Remove duplicates


def add_histogram_config_snippet():
    """Return the config snippet to add."""
    return """
# === Test loss histogram configuration ===
    c.plot_test_loss_histogram     = False  # Set to True to include test loss histogram in standard plotting
    c.plot_only_test_loss_histogram = False  # Set to True to skip all other plots and only generate test loss histogram
    c.test_loss_histogram_component = "loss_test"  # Which loss component to plot (e.g., "loss_test", "kl_test", "reco_test")
"""


def get_config_values_from_snippet():
    """Extract the expected config values from the snippet."""
    return {
        'plot_test_loss_histogram': False,
        'plot_only_test_loss_histogram': False,
        'test_loss_histogram_component': "loss_test"
    }


def parse_config_value(line):
    """Parse a config line and extract the parameter name and value."""
    # Handle different formats: c.param = value, c.param= value, etc.
    match = re.match(r'\s*c\.(\w+)\s*=\s*(.+?)(?:\s*#.*)?$', line.strip())
    if not match:
        return None, None
    
    param_name = match.group(1)
    value_str = match.group(2).strip()
    
    # Parse the value based on type
    if value_str.lower() in ['true', 'false']:
        value = value_str.lower() == 'true'
    elif value_str.startswith('"') and value_str.endswith('"'):
        value = value_str[1:-1]  # Remove quotes
    elif value_str.startswith("'") and value_str.endswith("'"):
        value = value_str[1:-1]  # Remove quotes
    else:
        # Try to evaluate as number or keep as string
        try:
            if '.' in value_str:
                value = float(value_str)
            else:
                value = int(value_str)
        except ValueError:
            value = value_str
    
    return param_name, value


def find_existing_config_values(content):
    """Find existing histogram config values in the file content."""
    lines = content.split('\n')
    existing_values = {}
    config_line_indices = {}
    
    target_params = {'plot_test_loss_histogram', 'plot_only_test_loss_histogram', 'test_loss_histogram_component'}
    
    for i, line in enumerate(lines):
        param_name, value = parse_config_value(line)
        if param_name in target_params:
            existing_values[param_name] = value
            config_line_indices[param_name] = i
    
    return existing_values, config_line_indices


def values_match(existing_values, expected_values):
    """Check if existing config values match expected values."""
    for param, expected_value in expected_values.items():
        if param not in existing_values:
            return False
        if existing_values[param] != expected_value:
            return False
    return True


def generate_config_line(param_name, value, comment=""):
    """Generate a properly formatted config line."""
    if isinstance(value, bool):
        value_str = str(value)
    elif isinstance(value, str):
        value_str = f'"{value}"'
    else:
        value_str = str(value)
    
    # Format with proper spacing
    line = f"    c.{param_name:<30} = {value_str}"
    if comment:
        line += f"  # {comment}"
    
    return line


def process_config_file(file_path, dry_run=True):
    """Add or update histogram config in a single file."""

    # Read file
    with open(file_path, "r") as f:
        content = f.read()

    # Get expected values from snippet
    expected_values = get_config_values_from_snippet()
    
    # Check if config already exists
    existing_values, config_line_indices = find_existing_config_values(content)
    
    if not existing_values:
        # No existing config - add it using original logic
        lines = content.split("\n")
        insert_idx = len(lines) - 1

        # Look for good insertion points
        for i, line in enumerate(lines):
            if any(pattern in line for pattern in ["annealing", "def set_config", "# ==="]):
                if "annealing" in line.lower():
                    insert_idx = i
                    break

        # Insert the config
        config_snippet = add_histogram_config_snippet()
        new_lines = lines[:insert_idx] + config_snippet.split("\n") + lines[insert_idx:]
        new_content = "\n".join(new_lines)
        
        if not dry_run:
            # Backup original
            backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(file_path, backup_path)

            # Write modified content
            with open(file_path, "w") as f:
                f.write(new_content)

            return True, "Added new histogram config (backup created)"
        else:
            return True, "Would add new histogram config"
    
    else:
        # Config exists - check if values match
        if values_match(existing_values, expected_values):
            return False, "Config exists with matching values - no changes needed"
        
        # Values don't match - update them
        lines = content.split('\n')
        
        # Define comments for each parameter
        comments = {
            'plot_test_loss_histogram': "Set to True to include test loss histogram in standard plotting",
            'plot_only_test_loss_histogram': "Set to True to skip all other plots and only generate test loss histogram", 
            'test_loss_histogram_component': 'Which loss component to plot (e.g., "loss_test", "kl_test", "reco_test")'
        }
        
        # Update existing lines
        for param_name, expected_value in expected_values.items():
            if param_name in config_line_indices:
                line_idx = config_line_indices[param_name]
                current_value = existing_values.get(param_name)
                
                if current_value != expected_value:
                    # Replace the line with new value
                    new_line = generate_config_line(param_name, expected_value, comments.get(param_name, ""))
                    lines[line_idx] = new_line
        
        new_content = '\n'.join(lines)
        
        if not dry_run:
            # Backup original
            backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(file_path, backup_path)

            # Write modified content
            with open(file_path, "w") as f:
                f.write(new_content)

            # Show what was changed
            changes = []
            for param_name, expected_value in expected_values.items():
                if param_name in existing_values and existing_values[param_name] != expected_value:
                    changes.append(f"{param_name}: {existing_values[param_name]} → {expected_value}")
            
            change_summary = ", ".join(changes) if changes else "values updated"
            return True, f"Updated config values: {change_summary} (backup created)"
        else:
            # Show what would be changed
            changes = []
            for param_name, expected_value in expected_values.items():
                if param_name in existing_values and existing_values[param_name] != expected_value:
                    changes.append(f"{param_name}: {existing_values[param_name]} → {expected_value}")
            
            if changes:
                change_summary = ", ".join(changes)
                return True, f"Would update: {change_summary}"
            else:
                return False, "Config exists with matching values"


def main():
    import sys

    print("🔧 Enhanced Test Loss Histogram Config Manager")
    print("=" * 50)
    print("Behavior:")
    print("  • No config exists → Adds new config")
    print("  • Config exists with matching values → No changes")
    print("  • Config exists with different values → Updates to snippet values")
    print("=" * 50)

    # Check if path provided as argument
    if len(sys.argv) > 1:
        search_path = sys.argv[1]
        if os.path.isdir(search_path):
            pattern = os.path.join(search_path, "**/*.py")
            config_files = glob.glob(pattern, recursive=True)
        else:
            print(f"Error: {search_path} is not a valid directory")
            return
    else:
        print("Searching for config files in current directory...")
        config_files = find_config_files()

    # Filter to likely config files
    config_files = [
        f for f in config_files if "config" in f.lower() and f.endswith(".py")
    ]

    if not config_files:
        print("❌ No config files found!")
        print("\nTry running with a specific path:")
        print("  python add_histogram_to_configs.py /path/to/your/workspaces/")
        return

    print(f"📁 Found {len(config_files)} config files:")
    for i, file_path in enumerate(config_files, 1):
        print(f"  {i:2d}. {file_path}")

    print("\n" + "=" * 50)

    # Dry run first
    print("🔍 Analyzing config files...")
    needs_update = []
    no_changes_needed = []

    for file_path in config_files:
        try:
            would_modify, reason = process_config_file(file_path, dry_run=True)
            if would_modify:
                needs_update.append(file_path)
                print(f"📝 {os.path.basename(file_path)}: {reason}")
            else:
                no_changes_needed.append(file_path)
                print(f"✅ {os.path.basename(file_path)}: {reason}")
        except Exception as e:
            print(f"❌ {os.path.basename(file_path)}: Error - {e}")

    print(f"\n📊 Summary:")
    print(f"  • {len(needs_update)} files need updating")
    print(f"  • {len(no_changes_needed)} files are already correct")

    if not needs_update:
        print("\n🎉 All files are already configured correctly!")
        return

    print(f"\n📝 {len(needs_update)} files will be modified.")
    response = input("Proceed with modifications? (y/N): ").lower().strip()

    if response != "y":
        print("❌ Aborted.")
        return

    # Actually modify files
    print("\n🚀 Updating files...")
    success_count = 0
    update_count = 0
    add_count = 0

    for file_path in needs_update:
        try:
            was_modified, message = process_config_file(file_path, dry_run=False)
            if was_modified:
                if "Added new" in message:
                    add_count += 1
                    print(f"➕ {os.path.basename(file_path)}: {message}")
                elif "Updated config" in message:
                    update_count += 1
                    print(f"🔄 {os.path.basename(file_path)}: {message}")
                else:
                    print(f"✅ {os.path.basename(file_path)}: {message}")
                success_count += 1
            else:
                print(f"⚠️  {os.path.basename(file_path)}: {message}")
        except Exception as e:
            print(f"❌ {os.path.basename(file_path)}: Error - {e}")

    print(f"\n🎉 Successfully processed {success_count} config files!")
    if add_count > 0:
        print(f"   ➕ Added config to {add_count} files")
    if update_count > 0:
        print(f"   🔄 Updated config in {update_count} files")
    print("\n📋 To enable histogram-only plotting in any project:")
    print("   c.plot_only_test_loss_histogram = True")
    print("\n💡 To plot different loss components:")
    print("   c.test_loss_histogram_component = 'kl_test'  # or 'reco_test', etc.")


if __name__ == "__main__":
    main()
