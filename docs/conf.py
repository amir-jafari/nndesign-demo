"""Sphinx configuration for the nndesigndemos documentation."""

import datetime
import importlib.metadata
import importlib.util
import os
import re
import sys

# -- Path setup --------------------------------------------------------------
# Make the package importable for autodoc without installing it.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _read_version():
    """Return the package version so the docs never drift from the package.

    setup.py in this checkout is the source of truth, so it is read first
    (without importing the package, which would pull in PyQt6). If setup.py
    is ever removed, fall back to the metadata of the installed distribution.
    """
    setup_py = os.path.join(ROOT, "setup.py")
    if os.path.exists(setup_py):
        with open(setup_py, encoding="utf-8") as f:
            match = re.search(r"version\s*=\s*['\"]v?([^'\"]+)['\"]", f.read())
        if match:
            return match.group(1)
    try:
        return importlib.metadata.version("nndesigndemos").lstrip("v")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


# -- Project information -----------------------------------------------------
project = "nndesigndemos"
author = "Amir Jafari, Martin Hagan, Pedro Uría, Xiao Qi"
copyright = f"{datetime.date.today().year}, {author}"
release = _read_version()
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc -----------------------------------------------------------------
# PyQt6 needs a display and a compiled Qt install; mock it so the docs build on CI.
# The numeric stack is mocked only when it is not installed in the build environment.
autodoc_mock_imports = [
    "PyQt6",
    # matplotlib's Qt canvases probe for a real Qt binding at import time,
    # which fails when PyQt6 is mocked, so mock them as well.
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_qt5agg",
]
for _mod in ("numpy", "scipy", "matplotlib"):
    if importlib.util.find_spec(_mod) is None:
        autodoc_mock_imports.append(_mod)

autoclass_content = "both"  # merge class and __init__ docstrings
autodoc_member_order = "bysource"
autosummary_generate = False

# -- Napoleon ----------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- MyST --------------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist"]

# -- Copy button -------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |> "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}
html_static_path = ["_static"]
html_title = f"{project} {release} documentation"
html_show_sourcelink = False
