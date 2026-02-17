from typing import TypeVar, Generic, Callable, List
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
    """Get the absolute output path using the default plots folder."""
    current = Path.cwd()
    plots_dir = "plots"

    while current != current.parent:
        candidate = current / plots_dir
        if candidate.exists():
            result = candidate / relative_path
            logger.debug(f"get_plot_path called with: {relative_path}")
            logger.debug(f"Current working directory: {current}")
            logger.debug(f"Returning plot path: {result}")
            return result
        current = current.parent

    result = Path.cwd() / plots_dir / relative_path
    logger.debug(f"get_plot_path called with: {relative_path}")
    logger.debug(f"Current working directory: {current}")
    logger.debug(f"Returning plot path: {result}")
    return result
