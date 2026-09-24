"""Medical Imaging QA: reproducible quality checks for NIfTI images and label maps."""

from .api import inspect_nifti, run_batch, validate_pair
from .rules import QARules
from .version import __version__

__all__ = ["QARules", "inspect_nifti", "run_batch", "validate_pair"]
