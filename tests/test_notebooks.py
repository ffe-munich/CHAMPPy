"""Tests for executing Jupyter notebooks end-to-end."""

import pytest
import subprocess
import tempfile
from pathlib import Path


@pytest.fixture
def notebooks_dir():
    """Get the notebooks directory."""
    return Path(__file__).parent.parent / "notebooks"


def run_notebook_with_nbconvert(notebook_path):
    """
    Execute a Jupyter notebook using nbconvert.
    
    Parameters:
    notebook_path (Path): Path to the notebook to execute.
    
    Raises:
    RuntimeError: If notebook execution fails.
    """
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as tmp:
        output_notebook = tmp.name
    
    try:
        result = subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=600",
                f"--output={output_notebook}",
                str(notebook_path),
            ],
            capture_output=True,
            text=True,
            timeout=660  # 11 minutes total timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(
                f"Notebook execution failed:\n{result.stdout}\n{result.stderr}"
            )
            
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Notebook execution timed out: {notebook_path}")
    finally:
        # Clean up temporary file
        Path(output_notebook).unlink(missing_ok=True)


@pytest.mark.integration
def test_demo_without_parameterization(notebooks_dir):
    """Test execution of 01_demo_without_parameterization.ipynb notebook."""
    notebook = notebooks_dir / "01_demo_without_parameterization.ipynb"
    assert notebook.exists(), f"Notebook not found: {notebook}"
    run_notebook_with_nbconvert(notebook)


@pytest.mark.integration
def test_demo_including_parameterization(notebooks_dir):
    """Test execution of 02_demo_including_parameterization.ipynb notebook."""
    notebook = notebooks_dir / "02_demo_including_parameterization.ipynb"
    assert notebook.exists(), f"Notebook not found: {notebook}"
    run_notebook_with_nbconvert(notebook)
