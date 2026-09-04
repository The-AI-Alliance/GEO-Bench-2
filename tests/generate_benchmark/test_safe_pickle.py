# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

import ast
import gzip
import importlib
import io
import pickle

import h5py
import numpy as np
import pytest

from geobench_v2.generate_benchmark.safe_pickle import (
    safe_pickle_load,
    safe_pickle_loads,
)


class Payload:
    """Stands in for the arbitrary code an attacker puts in a tampered file."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (open, (str(self.path), "w"))


def _numpy_scalar_reconstructor():
    """numpy.core.multiarray was renamed to numpy._core.multiarray in numpy 2."""
    for module in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            return importlib.import_module(module).scalar
        except (ImportError, AttributeError):
            continue
    pytest.skip("numpy scalar reconstructor not available")


class ObjectScalarGadget:
    """Smuggles a payload through numpy's object-dtype scalar reconstructor."""

    def __init__(self, inner: bytes):
        self.inner = inner

    def __reduce__(self):
        return (_numpy_scalar_reconstructor(), (np.dtype("O"), self.inner))


def test_plain_metadata_round_trips():
    attrs = {"label": 3, "bands": ["a", "b"], "meta": {"nested": (1, 2.5, None)}}
    assert safe_pickle_loads(pickle.dumps(attrs)) == attrs


@pytest.mark.parametrize(
    "value", [np.int64(3), np.int32(7), np.uint8(255), np.float32(1.5)]
)
def test_numpy_scalar_labels_round_trip(value):
    loaded = safe_pickle_loads(pickle.dumps({"label": value}))["label"]
    assert loaded == value
    assert loaded.dtype == value.dtype


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_code_execution_is_refused(tmp_path, protocol):
    """Every pickle protocol routes callables through the blocked find_class."""
    proof = tmp_path / "executed"
    data = pickle.dumps(Payload(proof), protocol=protocol)

    with pytest.raises(pickle.UnpicklingError):
        safe_pickle_loads(data)

    assert not proof.exists()


def test_object_dtype_gadget_is_refused(tmp_path):
    """The allowed numpy names must not re-enter pickle on their buffer."""
    proof = tmp_path / "executed"
    gadget = pickle.dumps(ObjectScalarGadget(pickle.dumps(Payload(proof))))

    with pytest.raises(pickle.UnpicklingError):
        safe_pickle_loads(gadget)

    assert not proof.exists()


def test_hdf5_attribute_round_trip(tmp_path):
    """Covers the exact forestnet/so2sat read path."""
    path = tmp_path / "sample.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs["pickle"] = str(pickle.dumps({"label": 12}))

    with h5py.File(path, "r") as f:
        attr_dict = safe_pickle_loads(ast.literal_eval(f.attrs["pickle"]))

    assert attr_dict["label"] == 12


def test_hdf5_attribute_payload_is_refused(tmp_path):
    proof = tmp_path / "executed"
    path = tmp_path / "tampered.hdf5"
    with h5py.File(path, "w") as f:
        f.attrs["pickle"] = str(pickle.dumps(Payload(proof)))

    with h5py.File(path, "r") as f, pytest.raises(pickle.UnpicklingError):
        safe_pickle_loads(ast.literal_eval(f.attrs["pickle"]))

    assert not proof.exists()


def test_gzipped_grid_dict_round_trip(tmp_path):
    """Covers the kuro_siwo read path."""
    grid = {"hexid": {"info": {"actid": 205, "aoiid": 1, "grid_id": "abc"}}}
    path = tmp_path / "grid.gz"
    with gzip.open(path, "wb") as f:
        pickle.dump(grid, f)

    with gzip.open(path, "rb") as f:
        assert safe_pickle_load(f) == grid


def test_gzipped_grid_dict_payload_is_refused(tmp_path):
    proof = tmp_path / "executed"
    path = tmp_path / "grid.gz"
    with gzip.open(path, "wb") as f:
        pickle.dump(Payload(proof), f)

    with gzip.open(path, "rb") as f, pytest.raises(pickle.UnpicklingError):
        safe_pickle_load(f)

    assert not proof.exists()


def test_file_object_overload_matches_bytes_overload():
    data = pickle.dumps({"label": 1})
    assert safe_pickle_load(io.BytesIO(data)) == safe_pickle_loads(data)
