# interlatent/__init__.py
try:  # pragma: no cover - defensive import for older Python versions
    from importlib import metadata as _md

    _dist_map = getattr(_md, "packages_distributions", lambda: {})()
    __version__ = _md.version(__name__) if _dist_map.get(__name__) else "0.0.dev"
except Exception:
    __version__ = "0.0.dev"
# Nothing else yet; keep root namespace clean
