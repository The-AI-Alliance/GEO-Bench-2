# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

import os
import shutil
from typing import Any

import pytest
import torchvision.datasets.utils
from pytest import MonkeyPatch
from torchgeo.datasets.utils import Path


def copy(url: str, root: Path, *args: Any, **kwargs: Any) -> None:
    os.makedirs(root, exist_ok=True)
    shutil.copy(url, root)


@pytest.fixture(autouse=True)
def download_url(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(torchvision.datasets.utils, "download_url", copy)
    monkeypatch.setattr("geobench_v2.datasets.base.download_url", copy)
