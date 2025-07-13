# interlatent/__init__.py
from importlib import metadata as _md

__version__ = _md.version(__name__) if _md.packages_distributions().get(__name__) else "0.0.dev"
# Nothing else yet; keep root namespace clean
