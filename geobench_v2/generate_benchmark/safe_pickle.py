# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Restricted unpickling for data files that come from third parties.

Several source datasets ship metadata as pickles, either as an HDF5 attribute or
as a standalone ``.gz`` file. Loading those with :func:`pickle.load` executes
whatever the file asks for, so a tampered dataset can run arbitrary code on the
machine generating the benchmark (GHSA-6rc6-6gmr-73w8).

The loaders here only reconstruct plain data: containers, scalars and strings.
Every pickle opcode that can reach a callable resolves its target through
:meth:`pickle.Unpickler.find_class`, so refusing lookups there removes the code
execution path while leaving legitimate metadata untouched.
"""

import io
import pickle
from typing import IO, Any

import numpy as np


def _safe_numpy_scalar(dtype: np.dtype, data: bytes) -> Any:
    """Rebuild a numpy scalar without numpy's object-dtype code path.

    ``numpy.core.multiarray.scalar`` deserializes its buffer with ``pickle`` when
    the dtype holds objects, which would smuggle a payload past the restrictions
    below. Numeric dtypes have no such path and are rebuilt from the raw buffer.

    Args:
        dtype: dtype of the scalar to rebuild.
        data: raw buffer holding the scalar value.

    Returns:
        The reconstructed numpy scalar.

    Raises:
        pickle.UnpicklingError: If the dtype can hold arbitrary Python objects.
    """
    if dtype.hasobject or dtype.kind in "OV":
        raise pickle.UnpicklingError(
            f"refusing to rebuild a numpy scalar with dtype {dtype!r}"
        )
    return np.frombuffer(data, dtype=dtype, count=1)[0]


# numpy metadata such as an ``np.int64`` label needs these two names. Both are
# mapped to local implementations so that unpickling never calls into numpy's
# own reconstructors.
_ALLOWED_GLOBALS: dict[tuple[str, str], Any] = {
    ("numpy", "dtype"): np.dtype,
    ("numpy.core.multiarray", "scalar"): _safe_numpy_scalar,
    ("numpy._core.multiarray", "scalar"): _safe_numpy_scalar,
}


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that resolves only the globals needed for plain metadata."""

    def find_class(self, module: str, name: str) -> Any:
        """Resolve a global, rejecting anything outside the allow list.

        Args:
            module: module the pickle wants to import from.
            name: attribute the pickle wants to resolve.

        Returns:
            The allowed replacement for ``module.name``.

        Raises:
            pickle.UnpicklingError: For every name outside the allow list.
        """
        try:
            return _ALLOWED_GLOBALS[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(
                f"refusing to load '{module}.{name}' from an untrusted pickle"
            ) from None


def safe_pickle_load(file: IO[bytes]) -> Any:
    """Load a pickle from a file object without allowing code execution.

    Args:
        file: binary file object positioned at the start of the pickle.

    Returns:
        The deserialized object.

    Raises:
        pickle.UnpicklingError: If the pickle tries to resolve any callable.
    """
    return _RestrictedUnpickler(file).load()


def safe_pickle_loads(data: bytes) -> Any:
    """Load a pickle from bytes without allowing code execution.

    Args:
        data: raw pickle bytes.

    Returns:
        The deserialized object.

    Raises:
        pickle.UnpicklingError: If the pickle tries to resolve any callable.
    """
    return safe_pickle_load(io.BytesIO(data))
