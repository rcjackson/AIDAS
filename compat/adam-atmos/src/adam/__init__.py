"""Compatibility shim for the former ``adam`` package.

``adam`` (distributed on PyPI as ``adam-atmos``) has been renamed to ``aidas``
(``aidas-atmos``). This module forwards the old import path to the new package
so that existing code keeps working, and warns that it will go away.
"""

import importlib
import pkgutil
import sys
import warnings

# Emit the deprecation notice *before* importing aidas: that import pulls in
# torch and pyart, which install their own warnings filters and would otherwise
# suppress this message.
warnings.warn(
    "The 'adam' package (adam-atmos) has been renamed to 'aidas' "
    "(aidas-atmos) as part of the move to the Argonne AI-Driven Adaptive "
    "Sampling System. 'import adam' still works through this compatibility "
    "shim, but it will be removed in a future release. Please install "
    "'aidas-atmos' and use 'import aidas' instead.",
    DeprecationWarning,
    stacklevel=2,
)

import aidas  # noqa: E402
from aidas import io, model, testing, triggering, util, vis  # noqa: E402,F401

__version__ = aidas.__version__
__author__ = aidas.__author__

# Import every aidas submodule, then alias each under the old ``adam.`` prefix
# so that ``import adam.io`` and ``from adam.triggering.halo_lidar import ...``
# resolve to the same module objects as their ``aidas.`` counterparts.
for _info in pkgutil.walk_packages(aidas.__path__, prefix="aidas."):
    importlib.import_module(_info.name)

for _name, _module in list(sys.modules.items()):
    if _name.startswith("aidas."):
        sys.modules["adam." + _name[len("aidas.") :]] = _module

del _info, _name, _module
