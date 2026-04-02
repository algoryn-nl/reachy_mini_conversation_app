"""Tests for the runtime vision settings endpoints."""

from types import SimpleNamespace
from typing import Any
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from reachy_mini_conversation_app import vision_settings
from reachy_mini_conversation_app.vision_settings import (
    mount_vision_routes,
    _get_head_tracker_name,
)


@dataclass
class FakeDeps:
    """Minimal stand-in for ToolDependencies used by vision settings."""

    vision_processor: Any = None


@pytest.fixture(autouse=True)
def _reset_module_state() -> Any:
    """Reset module-level globals between tests."""
    vision_settings._cached_vision_processor = None
    vision_settings._vision_initializing = False
    vision_settings._vision_init_error = None
    vision_settings._active_head_tracker = None
    yield
    vision_settings._cached_vision_processor = None
    vision_settings._vision_initializing = False
    vision_settings._vision_init_error = None
    vision_settings._active_head_tracker = None


# ---------- _get_head_tracker_name ----------


def test_get_head_tracker_name_no_camera() -> None:
    """Returns None when camera_worker is None."""
    assert _get_head_tracker_name(None) is None


def test_get_head_tracker_name_no_tracker() -> None:
    """Returns None when head_tracker is not set."""
    cw = SimpleNamespace(head_tracker=None)
    assert _get_head_tracker_name(cw) is None  # type: ignore[arg-type]


def test_get_head_tracker_name_returns_cached_name() -> None:
    """Returns the cached tracker name when tracker is active."""
    vision_settings._active_head_tracker = "yolo"
    cw = SimpleNamespace(head_tracker=MagicMock())
    assert _get_head_tracker_name(cw) == "yolo"  # type: ignore[arg-type]


# ---------- mount_vision_routes / GET /vision/status ----------


@pytest.fixture
def app_and_deps() -> tuple[Any, FakeDeps, SimpleNamespace]:
    """Create a test FastAPI app with vision routes mounted."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    deps = FakeDeps()
    cw = SimpleNamespace(head_tracker=None)
    app = FastAPI()
    mount_vision_routes(app, deps, cw)  # type: ignore[arg-type]
    return TestClient(app), deps, cw


def test_status_endpoint_defaults(app_and_deps: tuple[Any, FakeDeps, SimpleNamespace]) -> None:
    """GET /vision/status returns default state."""
    client, deps, cw = app_and_deps
    resp = client.get("/vision/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["head_tracker"] is None
    assert data["local_vision"] is False
    assert data["local_vision_initializing"] is False
    assert data["local_vision_error"] is None
    assert data["camera_enabled"] is True


def test_status_reflects_vision_processor(app_and_deps: tuple[Any, FakeDeps, SimpleNamespace]) -> None:
    """GET /vision/status shows local_vision=True when processor is set."""
    client, deps, _ = app_and_deps
    deps.vision_processor = MagicMock()
    resp = client.get("/vision/status")
    assert resp.json()["local_vision"] is True


# ---------- POST /vision/head-tracker ----------


def test_set_head_tracker_disable(app_and_deps: tuple[Any, FakeDeps, SimpleNamespace]) -> None:
    """Setting tracker to null disables head tracking."""
    client, _, cw = app_and_deps
    cw.head_tracker = MagicMock()
    resp = client.post("/vision/head-tracker", json={"tracker": None})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["head_tracker"] is None
    assert cw.head_tracker is None


def test_set_head_tracker_invalid_name(app_and_deps: tuple[Any, FakeDeps, SimpleNamespace]) -> None:
    """Invalid tracker names are rejected."""
    client, _, _ = app_and_deps
    resp = client.post("/vision/head-tracker", json={"tracker": "invalid"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_tracker"


def test_set_head_tracker_init_failure(app_and_deps: tuple[Any, FakeDeps, SimpleNamespace]) -> None:
    """Tracker initialization failures return a 500 error."""
    client, _, _ = app_and_deps
    with patch(
        "reachy_mini_conversation_app.vision.yolo_head_tracker.HeadTracker",
        side_effect=RuntimeError("GPU init failed"),
    ):
        resp = client.post("/vision/head-tracker", json={"tracker": "yolo"})

    assert resp.status_code == 500
    assert resp.json()["error"] == "init_failed"


def test_set_head_tracker_no_camera() -> None:
    """Returns error when camera is not available."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    deps = FakeDeps()
    app = FastAPI()
    mount_vision_routes(app, deps, None)  # type: ignore[arg-type]
    client = TestClient(app)

    resp = client.post("/vision/head-tracker", json={"tracker": "yolo"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "camera_disabled"


# ---------- POST /vision/local-vision ----------


def test_disable_local_vision(app_and_deps: tuple[Any, FakeDeps, SimpleNamespace]) -> None:
    """Disabling local vision clears deps but caches the processor."""
    client, deps, _ = app_and_deps
    fake_proc = MagicMock()
    deps.vision_processor = fake_proc

    resp = client.post("/vision/local-vision", json={"enabled": False})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["local_vision"] is False
    assert deps.vision_processor is None
    assert vision_settings._cached_vision_processor is fake_proc


def test_enable_local_vision_from_cache(app_and_deps: tuple[Any, FakeDeps, SimpleNamespace]) -> None:
    """Re-enabling local vision uses the cached processor instantly."""
    client, deps, _ = app_and_deps
    fake_proc = MagicMock()
    vision_settings._cached_vision_processor = fake_proc

    resp = client.post("/vision/local-vision", json={"enabled": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["local_vision"] is True
    assert deps.vision_processor is fake_proc


def test_enable_local_vision_missing_dependency(app_and_deps: tuple[Any, FakeDeps, SimpleNamespace]) -> None:
    """Returns error when torch/transformers not installed."""
    client, deps, _ = app_and_deps

    with patch("importlib.import_module", side_effect=ImportError("No module")):
        resp = client.post("/vision/local-vision", json={"enabled": True})

    assert resp.status_code == 400
    assert resp.json()["error"] == "missing_dependency"


def test_enable_local_vision_already_initializing(app_and_deps: tuple[Any, FakeDeps, SimpleNamespace]) -> None:
    """Returns initializing status when init is already in progress."""
    client, _, _ = app_and_deps
    vision_settings._vision_initializing = True

    resp = client.post("/vision/local-vision", json={"enabled": True})

    assert resp.status_code == 200
    data = resp.json()
    assert data["initializing"] is True


# ---------- Seeding from startup state ----------


def test_mount_seeds_tracker_name_from_yolo() -> None:
    """mount_vision_routes detects yolo tracker from module name."""
    from fastapi import FastAPI

    deps = FakeDeps()
    tracker = MagicMock()
    type(tracker).__module__ = "reachy_mini_conversation_app.vision.yolo_head_tracker"
    cw = SimpleNamespace(head_tracker=tracker)

    app = FastAPI()
    mount_vision_routes(app, deps, cw)  # type: ignore[arg-type]

    assert vision_settings._active_head_tracker == "yolo"


def test_mount_seeds_tracker_name_from_mediapipe() -> None:
    """mount_vision_routes detects mediapipe tracker from module name."""
    from fastapi import FastAPI

    deps = FakeDeps()
    tracker = MagicMock()
    type(tracker).__module__ = "reachy_mini_toolbox.vision"
    cw = SimpleNamespace(head_tracker=tracker)

    app = FastAPI()
    mount_vision_routes(app, deps, cw)  # type: ignore[arg-type]

    assert vision_settings._active_head_tracker == "mediapipe"


def test_mount_seeds_cached_processor() -> None:
    """mount_vision_routes caches existing vision processor."""
    from fastapi import FastAPI

    fake_proc = MagicMock()
    deps = FakeDeps(vision_processor=fake_proc)
    cw = SimpleNamespace(head_tracker=None)

    app = FastAPI()
    mount_vision_routes(app, deps, cw)  # type: ignore[arg-type]

    assert vision_settings._cached_vision_processor is fake_proc
