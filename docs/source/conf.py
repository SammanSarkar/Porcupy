# -- Project information -----------------------------------------------------
project = 'Porcupy'
copyright = '2025, Samman Sarkar'
author = 'Samman Sarkar'
release = '0.2.0'
version = '0.2'  # optional but good practice

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',          # Support for Markdown
    'sphinx.ext.autodoc',   # For docstrings (if any)
    'sphinx.ext.napoleon',  # For Google-style docstrings
]

# Allow both .rst and .md
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']
