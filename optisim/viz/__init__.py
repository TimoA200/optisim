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
                "Install with `pip install optisim` or `pip install rich`."
            ) from _terminal_import_error

try:
    from optisim.viz.web import WebVisualizer
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without web deps installed.
    _web_import_error = exc

    class WebVisualizer:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ModuleNotFoundError(
                "Web visualization requires the 'web' optional dependencies. "
                "Install with `pip install optisim[web]`."
            ) from _web_import_error

__all__ = ["MatplotlibVisualizer", "TerminalVisualizer", "WebVisualizer"]
