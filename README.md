# CHAMPPy
`CHAMPPy` (Charging and Mobility Profiles in Python) is a Python library to generate mobility and charging profiles for different types of electric vehicles including vans, trucks, busses and passanger cars. 

## Repo structure

<!-- TODO: Zu umfangreich? -->

```
CHAMPPy/
├── src/champpy/                    # Main package source code
│   ├── __init__.py
│   ├── core/                       # Core functionality
│   │   ├── __init__.py
│   │   ├── charging/               # Charging profile module
│   │   │   ├── __init__.py
│   │   │   ├── charging_model.py   # Model to generate charging profiles
│   │   │   └── charging_validation.py  # Charging validation & plotting
│   │   └── mobility/               # Mobility profile module
│   │       ├── __init__.py
│   │       ├── mobility_cleaning.py    # Data cleaning
│   │       ├── mobility_components.py  # Data components
│   │       ├── mobility_data.py        # Data structures
│   │       ├── mobility_model.py       # Model to generate profiles
│   │       ├── mobility_validation.py  # Validation & plotting
│   │       ├── parameterization.py     # Parameter extraction
│   ├── utils/                      # Utilities
│   │   ├── __init__.py
│   │   ├── data_utils.py           # Ddata helpers
│   │   ├── logging.py              # Logging configuration
│   │   └── time_utils.py           # Time utilities
|   └── data/
        ├── params_info.parquet     
        └── params.parquet
├── notebooks/                      # Jupyter notebooks
│   ├── 01_demo_without_parameterization.ipynb # Demo notebook 1
│   └── 02_demo_including_parameterization.ipynb # Demo notebook 2
├── scripts/                        # Python scripts
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── conftest.py                 # Pytest configuration & fixtures
│   ├── test_full.py                # Full pipeline tests
│   ├── test_light.py               # Light tests
│   ├── test_mob_classes.py         # Mobility classes tests
│   ├── test_notebooks.py           # Notebook execution tests
│   ├── test_utilities.py           # Utility tests
│   └── test_validation.py          # Validation tests
├── data/                           # Data directory
├── plots/                          # Generated plots (HTML files)
├── pyproject.toml                  # Project configuration
├── LICENSE                         # License file
├── README.md                       # This file
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `src/champpy/core/mobility/` | Mobility profile generation, cleaning & validation |
| `src/champpy/core/charging/` | Charging profile generation & validation |
| `src/champpy/utils/` | Utility functions (paths, logging, time) |
| `src/champpy/data/` | Existing model parameters that can be used |
| `notebooks/` | Interactive demos and examples |
| `tests/` | Unit and integration tests |
| `data/` | Example reference data |


## Installation

### Prerequisites
- Python 3.8 or higher (3.11+ recommended)
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
pip install -e ".[dev]"
```

### Install from PyPI

<!-- TODO: Add when published to PyPI -->
```bash
pip install champpy
```

## Usage

### Quick Start

For detailed examples, check out the interactive Jupyter notebooks in the `notebooks/` directory:

1. **[01_demo_without_parameterization.ipynb](notebooks/01_demo_without_parameterization.ipynb)**  
   Simple demo showing how to generate mobility and charging profiles using existing model parameters.

2. **[02_demo_including_parameterization.ipynb](notebooks/02_demo_including_parameterization.ipynb)**  
   Complete workflow including parameterization from raw data, model generation, and validation.





