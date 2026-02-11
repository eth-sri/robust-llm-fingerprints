"""Compatibility shim for the legacy ``src`` namespace.

The project package now lives under ``robust_fp``. Prefer importing
``robust_fp`` directly. ``import src`` will continue to work when running
from the repository checkout so existing scripts are not immediately broken.
"""

from importlib import import_module
import sys

_module = import_module("robust_fp")
sys.modules[__name__] = _module
