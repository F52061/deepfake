"""Legacy import compatibility only. New integration belongs in integration/.

The original implementation is archived unchanged. Alias the whole module so
existing callers and module-level monkeypatches keep using the same object.
"""
import importlib as _importlib
import sys as _sys

_legacy = _importlib.import_module(
    "legacy_integration.archive_20261017.snapshot.vit_module.vit_m2f2_detector"
)
_sys.modules[__name__] = _legacy
