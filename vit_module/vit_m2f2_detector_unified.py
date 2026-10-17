"""Legacy LLaVA import compatibility; new code must not depend on this model."""
import importlib as _importlib
import sys as _sys

_legacy = _importlib.import_module(
    "legacy_integration.archive_20261017.snapshot.vit_module.vit_m2f2_detector_unified"
)
_sys.modules[__name__] = _legacy
