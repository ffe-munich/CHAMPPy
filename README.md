# CHAMPPy

<!-- TODO: Add Badges from https://img.shields.io -->

CHAMPPy (Charging and Mobility Profiles in Python) is a Python library to generate synthetic mobility and charging profiles for different types of electric vehicles including vans, trucks, busses and passanger cars. 

<p align="center">
   <img src="docs/_static/graphical_abstract.svg" width="80%" alt="Graphical Abstract">
</p>

Road transport decarbonization requires realistic charging demand models across all vehicle classes. However, many existing studies and publicly available tools focus mainly on private passenger cars, while commercial electric vehicles such as vans and trucks are often underrepresented despite their major relevance for emissions and grid impacts. CHAMPPy is an open Python package that addresses this gap by generating synthetic driving and charging profiles for different EV types, including commercial fleets. The model combines a Markov chain to represent vehicle locations over time with beta-distributed journey speeds, from which trip distances are derived, and uses dedicated algorithms to generate mobility and charging behavior. An optional clustering approach increases profile heterogeneity and is particularly useful when analyzing individual profiles.

CHAMPPy supports two workflows:

1. Use existing parameters to quickly generate drving and charging profiles with user-defined settings (e.g., simulation period, number of vehicles, charging power, battery capacity).
2. Re-parameterize the model with custom reference data (e.g., other countries, fleet compositions, or vehicle classes). Generate drving and charging profiles from your model parameters.

## Links

* Documentation: TODO
* Source code: [https://github.com/ffe-munich/CHAMPPy](https://github.com/ffe-munich/CHAMPPy)
* PyPI releases: TODO
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

<!-- docs-installation-start -->

### Prerequisites
- Python 3.13 or higher
- pip

### Install from source

```bash
# Clone the repository
git clone https://github.com/ffe-munich/CHAMPPy.git
cd CHAMPPy

# Create a virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.\.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install the package in development mode
pip install .
```

### Install from PyPI

```bash
# Create a virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.\.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

pip install champpy
```
<!-- docs-installation-end -->

<!-- sphinx-exclude-start -->
## Examples

For detailed examples, check out the interactive Jupyter notebooks in the `notebooks/` directory:

1. **[01_demo_without_parameterization.ipynb](notebooks/01_demo_without_parameterization.ipynb)**  
   Simple demo showing how to generate mobility and charging profiles using existing model parameters.

2. **[02_demo_including_parameterization.ipynb](notebooks/02_demo_including_parameterization.ipynb)**  
   Complete workflow including parameterization from raw data, model generation, and validation.
<!-- sphinx-exclude-end -->





