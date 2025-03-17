# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('..')))

# project_root = os.path.abspath('..')  # Adjust based on your actual structure
# package_dir = os.path.join(project_root, 'bead')

# sys.path.insert(0, project_root)
# sys.path.insert(0, package_dir)
# sys.path.insert(0, os.path.join(package_dir, 'src'))

# conf.py
autodoc_mock_imports = []  # Remove any mocked dependencies if not needed
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True
}
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'bead'
copyright = '2025, Pratik Jawahar'
author = 'Pratik Jawahar'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    'sphinx.ext.napoleon',
    "sphinx.ext.githubpages",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**tests**']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
