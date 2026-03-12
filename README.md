# CHAMPPy

[![PyPI version](https://img.shields.io/pypi/v/champpy.svg)](https://pypi.org/project/champpy/)
[![Python versions](https://img.shields.io/pypi/pyversions/champpy.svg)](https://pypi.org/project/champpy/)
[![PyPI downloads](https://img.shields.io/pypi/dm/champpy.svg)](https://pypi.org/project/champpy/)
[![License](https://img.shields.io/pypi/l/champpy.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://champpy.readthedocs.io)

CHAMPPy (Charging and Mobility Profiles in Python) is a Python library to generate synthetic mobility and charging profiles for different types of electric vehicles (EVs) including vans, trucks, busses and passanger cars. 

<p align="center">
   <img src="https://raw.githubusercontent.com/ffe-munich/CHAMPPy/main/data/graphical_abstract.svg" width="80%" alt="Graphical Abstract">
</p>

Road transport decarbonization requires realistic charging demand models across all vehicle classes. However, most existing studies and publicly available tools focus on private passenger cars. Commercial electric vehicles such as vans and trucks are often underrepresented despite their major relevance for emissions and grid impacts. CHAMPPy is an open Python package that addresses this gap by generating synthetic driving and charging profiles for different EV types, including commercial fleets. The model combines a Markov chain to represent vehicle locations over time with beta-distributed journey speeds, from which trip distances are derived, and uses dedicated algorithms to generate mobility and charging profiles. An optional clustering approach increases profile heterogeneity and is particularly useful when analyzing individual profiles.

🛠️ CHAMPPy supports two workflows:

1. 🚀 **Light:** Use existing parameters to quickly generate drving and charging profiles with user-defined settings (e.g., simulation period, number of vehicles, charging power, battery capacity).
2. 🧪 **Full:** Re-parameterize the model with custom reference data (e.g. driving data for other countries, vehicle classes, or fleets). Afterwards, you can generate drving and charging profiles from your model parameters.

## Links

* Documentation: [https://champpy.readthedocs.io](https://champpy.readthedocs.io/en/latest/)
* Source code: [https://github.com/ffe-munich/CHAMPPy](https://github.com/ffe-munich/CHAMPPy)
* PyPI releases: [https://pypi.org/project/champpy/](https://pypi.org/project/champpy/)
* License: [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT)

## Authors

CHAMPPy has been developed by [Florian Biedenbach](https://github.com/FloBieWan) (lead), Valentin Preis und [Daniel Godin](https://github.com/DaniGodin).

## Repo structure

```
CHAMPPy/
├── src/champpy/                        # Main package source code
│   ├── __init__.py
│   ├── core/                           # Core functionality
│   │   ├── __init__.py
│   │   ├── charging/                   # Charging profile module
│   │   │   ├── __init__.py
│   │   │   ├── charging_model.py       # Model to generate charging profiles
│   │   │   └── charging_validation.py  # Charging validation & plotting
│   │   └── mobility/                   # Mobility profile module
│   │       ├── __init__.py
│   │       ├── mobility_cleaning.py    # Data cleaning
│   │       ├── mobility_components.py  # Data components
│   │       ├── mobility_data.py        # Data structures
│   │       ├── mobility_model.py       # Model to generate profiles
│   │       ├── mobility_validation.py  # Validation & plotting
│   │       └── parameterization.py     # Parameter extraction
│   ├── utils/                          # Utilities
│   │   ├── __init__.py
│   │   ├── data_utils.py               # Ddata helpers
│   │   ├── logging.py                  # Logging configuration
│   │   └── time_utils.py               # Time utilities
|   └── data/                       
│       ├── params_info.parquet         # Info about existing model parameters
│       └── params.parquet              # Existing model parameters
├── notebooks/                          # Jupyter notebooks
│   ├── 01_demo_without_parameterization.ipynb # Demo notebook 1
│   └── 02_demo_including_parameterization.ipynb # Demo notebook 2
├── scripts/                            # Python scripts
├── tests/                              # Test suite
├── data/                               # Data directory
├── plots/                              # Generated plots (HTML files)
├── pyproject.toml                      # Project configuration
├── LICENSE                             # License file
└──  README.md                          # This file
```

## Installation

To install Champy on Windows, follow the step below. For installation on Linux/Mac, please check the [installation documentation on Read the Docs](https://champpy.readthedocs.io/en/latest/installation.html).

### Prerequisites
- Python 3.11 or higher
- pip

### Install from source on windows

```bash
# Clone the repository
git clone https://github.com/ffe-munich/CHAMPPy.git
cd CHAMPPy

# Create a virtual environment
py -m venv .venv

# Activate virtual environment
.\.venv\Scripts\activate


# Install the package
pip install .
```

### Install from PyPI on windows

```bash
# Create a virtual environment
py -m venv .venv

# Activate virtual environment
.\.venv\Scripts\activate

pip install champpy
```

<!-- sphinx-exclude-start -->
## Examples

To get started, check out the interactive Jupyter notebooks in the `notebooks/` directory:

1. **[01_demo_without_parameterization.ipynb](notebooks/01_demo_without_parameterization.ipynb)**  
   Simple demo showing how to generate mobility and charging profiles using existing model parameters.

2. **[02_demo_including_parameterization.ipynb](notebooks/02_demo_including_parameterization.ipynb)**  
   Complete workflow including parameterization from reference data, model generation, and validation.
<!-- sphinx-exclude-end -->





