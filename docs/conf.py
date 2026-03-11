# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CHAMPPy"
copyright = "2026, FfE Munich"
author = "Florian Biedenbach"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
import shutil
import tomllib
from pathlib import Path

sys.path.insert(0, os.path.abspath("../"))

# Read version from pyproject.toml
with open(os.path.join(os.path.dirname(__file__), "..", "pyproject.toml"), "rb") as f:
    _pyproject = tomllib.load(f)
release = _pyproject["project"]["version"]
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "myst_nb",
]

autodoc_default_options = {
    "members": True,
    "member-order": "groupwise",  # Group by type: attributes/properties first, then methods
    "undoc-members": False,
    "private-members": False,
}

# Napoleon settings for better docstring rendering
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

autosummary_generate = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST-NB configuration
nb_execution_mode = "off"  # Don't execute notebooks during build

# Suppress warnings for README notebook links
suppress_warnings = ["myst.xref_missing"]

language = "[en]"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "CHAMPPy Documentation"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_with_keys": True,
}
html_css_files = ["custom.css"]


def sync_notebooks_from_repo(app):
    """Synchronize notebooks from repo-level notebooks/ into docs/notebooks/."""
    repo_root = os.path.dirname(app.srcdir)
    src_dir = os.path.join(repo_root, "notebooks")
    dst_dir = os.path.join(app.srcdir, "notebooks")

    if not os.path.isdir(src_dir):
        return

    os.makedirs(dst_dir, exist_ok=True)

    src_files = {
        f
        for f in os.listdir(src_dir)
        if f.endswith(".ipynb") and (f.startswith("01_demo") or f.startswith("02_demo"))
    }
    dst_files = {f for f in os.listdir(dst_dir) if f.endswith(".ipynb")}

    # Remove stale notebook copies
    for stale in dst_files - src_files:
        os.remove(os.path.join(dst_dir, stale))

    # Copy/update current notebooks
    for nb in src_files:
        shutil.copy2(os.path.join(src_dir, nb), os.path.join(dst_dir, nb))

def setup(app):
    app.connect("builder-inited", sync_notebooks_from_repo)
