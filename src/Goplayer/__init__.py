from .goenv import GoEnv

__all__ = ["GoEnv"]

# GUI components are optional (install with: uv sync --extra gui)
try:
    from .goboard import GoBoard  # type: ignore
except Exception:
    pass
else:
    __all__.append("GoBoard")
