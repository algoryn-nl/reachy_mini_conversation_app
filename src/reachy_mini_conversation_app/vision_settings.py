"""FastAPI routes for runtime vision settings (head tracker + local vision).

Mounted on the headless settings app so users can toggle vision features
from the embedded web UI without restarting the daemon.
"""

import logging
import threading
from typing import Any, Optional

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.camera_worker import CameraWorker

logger = logging.getLogger(__name__)

# Module-level cache so the vision processor survives disable/re-enable cycles.
_cached_vision_processor: Any = None
_vision_initializing = False
_vision_init_error: Optional[str] = None

# Track which head tracker backend is active (for status reporting).
_active_head_tracker: Optional[str] = None


def _get_head_tracker_name(camera_worker: Optional[CameraWorker]) -> Optional[str]:
    """Return the active head tracker backend name, or None."""
    global _active_head_tracker
    if camera_worker is None or camera_worker.head_tracker is None:
        _active_head_tracker = None
        return None
    return _active_head_tracker


def mount_vision_routes(
    app: Any,
    deps: ToolDependencies,
    camera_worker: Optional[CameraWorker],
) -> None:
    """Register vision settings endpoints on a FastAPI app."""
    try:
        from fastapi import Request
        from fastapi.responses import JSONResponse
    except Exception:
        return

    global _cached_vision_processor, _vision_initializing, _vision_init_error, _active_head_tracker

    # Seed tracker name from whatever was set at startup
    if camera_worker is not None and camera_worker.head_tracker is not None:
        tracker = camera_worker.head_tracker
        cls_name = type(tracker).__module__
        if "yolo" in cls_name:
            _active_head_tracker = "yolo"
        else:
            _active_head_tracker = "mediapipe"

    # Seed cached processor from startup
    if deps.vision_processor is not None:
        _cached_vision_processor = deps.vision_processor

    # ------------------------------------------------------------------
    # GET /vision/status
    # ------------------------------------------------------------------
    @app.get("/vision/status")
    def _vision_status() -> JSONResponse:
        return JSONResponse({
            "head_tracker": _get_head_tracker_name(camera_worker),
            "local_vision": deps.vision_processor is not None,
            "local_vision_initializing": _vision_initializing,
            "local_vision_error": _vision_init_error,
            "camera_enabled": camera_worker is not None,
        })

    # ------------------------------------------------------------------
    # POST /vision/head-tracker
    # ------------------------------------------------------------------
    @app.post("/vision/head-tracker")
    async def _set_head_tracker(request: Request) -> JSONResponse:
        global _active_head_tracker

        if camera_worker is None:
            return JSONResponse(
                {"ok": False, "error": "camera_disabled", "detail": "Camera is not enabled. Start with camera support."},
                status_code=400,
            )

        try:
            body = await request.json()
        except Exception:
            body = {}
        tracker_name = body.get("tracker")  # "yolo", "mediapipe", or null/empty

        if not tracker_name:
            camera_worker.head_tracker = None
            _active_head_tracker = None
            logger.info("Head tracker disabled via settings UI")
            return JSONResponse({"ok": True, "head_tracker": None})

        if tracker_name not in ("yolo", "mediapipe"):
            return JSONResponse(
                {"ok": False, "error": "invalid_tracker", "detail": f"Unknown tracker: {tracker_name}"},
                status_code=400,
            )

        try:
            if tracker_name == "yolo":
                from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker
                new_tracker = HeadTracker()
            else:
                from reachy_mini_toolbox.vision import HeadTracker  # type: ignore[no-redef]
                new_tracker = HeadTracker()

            camera_worker.head_tracker = new_tracker
            _active_head_tracker = tracker_name
            logger.info("Head tracker switched to %s via settings UI", tracker_name)
            return JSONResponse({"ok": True, "head_tracker": tracker_name})

        except ImportError:
            extra = "yolo_vision" if tracker_name == "yolo" else "mediapipe_vision"
            return JSONResponse(
                {
                    "ok": False,
                    "error": "missing_dependency",
                    "detail": f"Install the required extra: pip install '.[{extra}]'",
                },
                status_code=400,
            )
        except Exception as e:
            logger.exception("Failed to initialize %s head tracker", tracker_name)
            return JSONResponse(
                {"ok": False, "error": "init_failed", "detail": str(e)},
                status_code=500,
            )

    # ------------------------------------------------------------------
    # POST /vision/local-vision
    # ------------------------------------------------------------------
    @app.post("/vision/local-vision")
    async def _set_local_vision(request: Request) -> JSONResponse:
        global _cached_vision_processor, _vision_initializing, _vision_init_error

        try:
            body = await request.json()
        except Exception:
            body = {}
        enabled = body.get("enabled", False)

        if not enabled:
            # Disable: clear from deps but keep cached
            if deps.vision_processor is not None:
                _cached_vision_processor = deps.vision_processor
            deps.vision_processor = None
            logger.info("Local vision disabled via settings UI")
            return JSONResponse({"ok": True, "local_vision": False})

        # Enable: use cached processor if available
        if _cached_vision_processor is not None:
            deps.vision_processor = _cached_vision_processor
            logger.info("Local vision re-enabled from cache via settings UI")
            return JSONResponse({"ok": True, "local_vision": True})

        # Need to initialize — check if already in progress
        if _vision_initializing:
            return JSONResponse({"ok": True, "local_vision": False, "initializing": True})

        # Check that the extra is installed before spawning a thread
        try:
            import importlib
            importlib.import_module("torch")
            importlib.import_module("transformers")
        except ImportError:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "missing_dependency",
                    "detail": "Install the required extra: pip install '.[local_vision]'",
                },
                status_code=400,
            )

        # Start background initialization
        _vision_initializing = True
        _vision_init_error = None

        def _init_vision() -> None:
            global _cached_vision_processor, _vision_initializing, _vision_init_error
            try:
                from reachy_mini_conversation_app.vision.processors import initialize_vision_processor
                processor = initialize_vision_processor()
                _cached_vision_processor = processor
                deps.vision_processor = processor
                logger.info("Local vision initialized in background via settings UI")
            except Exception as e:
                logger.exception("Background vision initialization failed")
                _vision_init_error = str(e)
            finally:
                _vision_initializing = False

        thread = threading.Thread(target=_init_vision, daemon=True, name="vision-init")
        thread.start()

        return JSONResponse({"ok": True, "local_vision": False, "initializing": True})
