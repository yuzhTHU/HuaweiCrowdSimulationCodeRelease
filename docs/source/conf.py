import os
import sys
# 将项目根目录加入路径，以便 Sphinx 找到源码
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
    'sphinx.ext.autodoc',      # 自动从 docstring 生成文档
    'sphinx.ext.napoleon',     # 支持 Google/NumPy 风格的 docstring
    'sphinx.ext.viewcode',     # 在文档中添加“查看源代码”链接
    # 'sphinx.ext.todo',
    'myst_parser',             # 支持 Markdown (.md) 文件
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_title = "人群模拟仿真系统 API 文档"

# 自动生成配置
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}