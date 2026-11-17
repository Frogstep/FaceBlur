"""Shared test setup.

Test video: _DSG0777.mp4 in the project folder (git-ignored, so it is only on the
machines that have it); FACEBLUR_TEST_VIDEO points to another file.

End-to-end tests (real models, GPU, minutes) are marked `e2e` and run only when
selected:  .venv/Scripts/python -m pytest -m e2e tests/server
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEST_VIDEO = Path(os.environ.get("FACEBLUR_TEST_VIDEO") or ROOT / "_DSG0777.mp4")


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: real pipeline on the GPU (slow); run with -m e2e")


def pytest_collection_modifyitems(config, items):
    if "e2e" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="end-to-end test: run with -m e2e")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def test_video() -> Path:
    if not TEST_VIDEO.is_file():
        pytest.skip(f"test video not found: {TEST_VIDEO} (set FACEBLUR_TEST_VIDEO)")
    return TEST_VIDEO
