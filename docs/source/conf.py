import os
import sys
# Add the project root to the path so Sphinx can find the source code.
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CrowdSimulation'
copyright = '2025, FIB-Lab'
author = 'FIB-Lab'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

autodoc_mock_imports = [
    'torch', 
    'numpy', 
    'pandas', 
    'scipy', 
    'matplotlib', 
    'seaborn', 
    'PIL', 
    'tqdm', 
    'setproctitle', 
    'fastapi',
    'uvicorn',
    'websockets'
]

extensions = [
    'sphinx.ext.autodoc',      # Automatically generate docs from docstrings.
    'sphinx.ext.napoleon',     # Support Google/NumPy-style docstrings.
    'sphinx.ext.viewcode',     # Add "view source code" links in the docs.
    # 'sphinx.ext.todo',
    'myst_parser',             # Support Markdown (.md) files.
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_title = "Crowd Simulation System API Documentation"

# Auto-generated documentation configuration.
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
