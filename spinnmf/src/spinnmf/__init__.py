from importlib.metadata import PackageNotFoundError, version

from .autok import fit_spin_nmf_autok
from .model import fit_spin_nmf

__all__ = ["fit_spin_nmf", "fit_spin_nmf_autok"]

try:
    __version__ = version("spinnmf")
except PackageNotFoundError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.0.0"
