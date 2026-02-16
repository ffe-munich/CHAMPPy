from typing import TypeVar, Generic, Callable, List
import tomli
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
T = TypeVar("T")

class Event(Generic[T]):
    """
    Simple Observer/Event base class for use in the Champpy project
    e.g. in MobData, LogBooks, Vehicles, Clusters for subclasses of MobData to trigger events on data changes.
    """
    def __init__(self):
        self._observers: List[Callable[[T], None]] = []

    def add_observer(self, fn: Callable[[T], None]) -> None:
        if fn not in self._observers:
            self._observers.append(fn)

    def delete_observer(self, fn: Callable[[T], None]) -> None:
        if fn in self._observers:
            self._observers.remove(fn)

    def trigger(self, data: T) -> None:
        for funct in self._observers:
            funct(data)

def get_plot_path(relative_path: Path) -> Path:
    """Get the absolute output path by reading configuration from pyproject.toml.
    
    Searches for pyproject.toml starting from the current working directory first (for notebooks),
    then from the module location (for scripts).
    """
    # Try to find project root starting from current working directory (for Jupyter notebooks)
    current = Path.cwd()
    project_root = None
    
    logger.debug(f"get_plot_path called with: {relative_path}")
    logger.debug(f"Current working directory: {current}")
    
    # Search upwards from current working directory
    while current != current.parent:
        if (current / 'pyproject.toml').exists():
            project_root = current
            logger.debug(f"Found project root from cwd: {project_root}")
            break
        current = current.parent
    
    # If not found from cwd, try from module location (for installed packages)
    if not project_root:
        current = Path(__file__).resolve().parent
        logger.debug(f"Searching from module location: {current}")
        while current != current.parent:
            if (current / 'pyproject.toml').exists():
                project_root = current
                logger.debug(f"Found project root from module: {project_root}")
                break
            current = current.parent
    
    # If pyproject.toml found, read plot_dir configuration
    if project_root and tomli:
        try:
            with open(project_root / 'pyproject.toml', 'rb') as f:
                config = tomli.load(f)
                plot_dir = config.get('tool', {}).get('champpy', {}).get('plot_dir', 'plots')
                result = project_root / plot_dir / relative_path
                logger.debug(f"Returning plot path: {result}")
                return result
        except Exception as e:
            logger.debug(f"Could not read pyproject.toml: {e}")
    
    # Fallback: if installed as package, write to current working directory
    logger.debug("Using current working directory as output path (package installation mode)")
    return Path.cwd() / relative_path