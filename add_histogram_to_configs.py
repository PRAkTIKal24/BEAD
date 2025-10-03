#!/usr/bin/env python3
"""
Simple script to add test loss histogram config to multiple config files.

Usage:
1. Save this script in your BEAD repository root
2. Run: python add_histogram_to_configs.py /path/to/your/workspace/configs/
   OR just run it and it will search common locations automatically
"""

import os
import glob
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
            "*/*/*/config/*.py"
        ]
    
    config_files = []
    for pattern in search_paths:
        found = glob.glob(pattern, recursive=True)
        config_files.extend([f for f in found if os.path.isfile(f)])
    
    return list(set(config_files))  # Remove duplicates

def add_histogram_config_snippet():
    """Return the config snippet to add."""
    return '''
# === Test loss histogram configuration ===
    c.plot_test_loss_histogram     = False  # Set to True to include test loss histogram in standard plotting
    c.plot_only_test_loss_histogram = False  # Set to True to skip all other plots and only generate test loss histogram
    c.test_loss_histogram_component = "loss_test"  # Which loss component to plot (e.g., "loss_test", "kl_test", "reco_test")
'''

def process_config_file(file_path, dry_run=True):
    """Add histogram config to a single file."""
    
    # Read file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already has histogram config
    if any(marker in content for marker in ['plot_test_loss_histogram', 'plot_only_test_loss_histogram']):
        return False, "Already has histogram config"
    
    # Find insertion point (before annealing config or at end of function)
    lines = content.split('\n')
    insert_idx = len(lines) - 1
    
    # Look for good insertion points
    for i, line in enumerate(lines):
        if any(pattern in line for pattern in ['annealing', 'def set_config', '# ===']):
            if 'annealing' in line.lower():
                insert_idx = i
                break
    
    # Insert the config
    config_snippet = add_histogram_config_snippet()
    new_lines = lines[:insert_idx] + config_snippet.split('\n') + lines[insert_idx:]
    new_content = '\n'.join(new_lines)
    
    if not dry_run:
        # Backup original
        backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(file_path, backup_path)
        
        # Write modified content
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        return True, f"Modified (backup created)"
    else:
        return True, "Would be modified"

def main():
    import sys
    
    print("🔧 Test Loss Histogram Config Adder")
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
    config_files = [f for f in config_files if 'config' in f.lower() and f.endswith('.py')]
    
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
    print("🔍 Checking which files need updating...")
    needs_update = []
    
    for file_path in config_files:
        try:
            would_modify, reason = process_config_file(file_path, dry_run=True)
            if would_modify:
                needs_update.append(file_path)
                print(f"✅ {os.path.basename(file_path)}: {reason}")
            else:
                print(f"⏭️  {os.path.basename(file_path)}: {reason}")
        except Exception as e:
            print(f"❌ {os.path.basename(file_path)}: Error - {e}")
    
    if not needs_update:
        print("\n🎉 All files already have histogram config!")
        return
    
    print(f"\n📝 {len(needs_update)} files need updating.")
    response = input("Proceed with modifications? (y/N): ").lower().strip()
    
    if response != 'y':
        print("❌ Aborted.")
        return
    
    # Actually modify files
    print("\n🚀 Updating files...")
    success_count = 0
    
    for file_path in needs_update:
        try:
            was_modified, message = process_config_file(file_path, dry_run=False)
            if was_modified:
                print(f"✅ {os.path.basename(file_path)}: {message}")
                success_count += 1
            else:
                print(f"⚠️  {os.path.basename(file_path)}: {message}")
        except Exception as e:
            print(f"❌ {os.path.basename(file_path)}: Error - {e}")
    
    print(f"\n🎉 Successfully updated {success_count} config files!")
    print("\n📋 To enable histogram-only plotting in any project:")
    print("   c.plot_only_test_loss_histogram = True")
    print("\n💡 To plot different loss components:")
    print("   c.test_loss_histogram_component = 'kl_test'  # or 'reco_test', etc.")

if __name__ == "__main__":
    main()