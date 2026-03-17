"""Visualization backends."""

from optisim.viz.matplotlib_viz import MatplotlibVisualizer

try:
    from optisim.viz.terminal import TerminalVisualizer
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without Rich installed.
    _terminal_import_error = exc

    class TerminalVisualizer:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ModuleNotFoundError(
                "Terminal visualization requires the 'rich' dependency. "
                "Install optisim with `pip install -e .` or add `rich` explicitly."
            ) from _terminal_import_error

__all__ = ["MatplotlibVisualizer", "TerminalVisualizer"]
