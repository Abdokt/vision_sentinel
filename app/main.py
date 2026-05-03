import asyncio
import time
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.camera import StreamReader
from app.config import settings
from app.counter import LineCrossCounter
from app.detector import Detector, DetectionResult
from app.monitor import get_system_metrics
from app.watchdog import WindowsDeviceWatchdog


# ── rate limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── shared state ───────────────────────────────────────────────────────────────
# These hold the running cameras, detectors, and result queues.
# Populated during lifespan startup, read by WebSocket handlers.
# Keys are present only while the camera is enabled — absence means disabled.
cameras: dict[str, StreamReader] = {}
detectors: dict[str, Detector] = {}
result_queues: dict[str, asyncio.Queue] = {}
counters: dict[str, LineCrossCounter] = {}
models: dict = {}   # YOLO model instances — kept loaded across enable/disable cycles
watchdog: WindowsDeviceWatchdog | None = None


# ── camera lifecycle helpers ───────────────────────────────────────────────────

async def _start_cam(cam_key: str, cam_idx: int) -> None:
    """Load model (if needed), create and start camera + detector."""
    from ultralytics import YOLO

    cam_name = settings.camera_0_name if cam_idx == 0 else settings.camera_1_name
    dev_name = (
        settings.camera_0_device_name if cam_idx == 0
        else settings.camera_1_device_name
    )
    loop = asyncio.get_running_loop()

    # Model loading is blocking — run in a thread so the event loop stays free
    if cam_key not in models:
        logger.info(f"Loading YOLO11 model for {cam_key}...")

        def _load():
            m = YOLO(settings.model_path)
            m.to(settings.device)
            return m

        models[cam_key] = await loop.run_in_executor(None, _load)
        logger.success(f"[{cam_key}] Model loaded on {settings.device}.")

    cameras[cam_key] = StreamReader(
        source=settings.get_camera_source(cam_idx),
        name=cam_name,
        width=settings.camera_width,
        height=settings.camera_height,
        backend=settings.camera_backend,
    ).start()

    result_queues[cam_key] = asyncio.Queue(maxsize=2)

    if cam_key == "cam0":
        counters["cam0"] = LineCrossCounter(
            camera_name=cam_name,
            max_occupancy=settings.max_occupancy,
        )
        counters["cam0"].set_tripwire(0.0, 0.5, 1.0, 0.5, entry_direction="positive")

    detectors[cam_key] = Detector(
        stream=cameras[cam_key],
        model=models[cam_key],
        confidence=settings.confidence_threshold,
        device=settings.device,
        imgsz=settings.inference_size,
        frame_skip=settings.frame_skip,
        result_queue=result_queues[cam_key],
        loop=loop,
        counter=counters.get(cam_key),
    ).start()

    if watchdog:
        watchdog.watch(dev_name, cameras[cam_key])

    logger.info(f"[{cam_key}] Started on camera index {cam_idx}.")


def _stop_cam(cam_key: str) -> None:
    """Stop and clean up detector + camera. Model stays loaded for fast re-enable."""
    if cam_key in detectors:
        detectors[cam_key].stop()
        del detectors[cam_key]
    if cam_key in cameras:
        cameras[cam_key].stop()
        del cameras[cam_key]
    if cam_key in result_queues:
        del result_queues[cam_key]
    logger.info(f"[{cam_key}] Stopped.")


# ── lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup: starts cameras and detectors for enabled camera slots.
    Runs on shutdown: stops everything cleanly.
    """
    global watchdog
    logger.info("Vision Sentinel starting up...")

    if settings.enable_device_watchdog:
        watchdog = WindowsDeviceWatchdog(check_interval=2.0)
        watchdog.start()

    if settings.camera_0_enabled:
        await _start_cam("cam0", 0)
    else:
        logger.info("[cam0] Disabled by config (CAMERA_0_ENABLED=false) — skipping.")

    if settings.camera_1_enabled:
        await _start_cam("cam1", 1)
    else:
        logger.info("[cam1] Disabled by config (CAMERA_1_ENABLED=false) — skipping.")

    logger.success("Vision Sentinel ready.")
    yield  # server runs here

    # ── shutdown ───────────────────────────────────────────────────────
    logger.info("Vision Sentinel shutting down...")
    if watchdog:
        watchdog.stop()
    for key in list(detectors.keys()):
        detectors[key].stop()
    for key in list(cameras.keys()):
        cameras[key].stop()
    logger.info("Shutdown complete.")


# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vision Sentinel",
    description="Real-time multi-camera object detection and tracking.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── API key check ──────────────────────────────────────────────────────────────
def verify_api_key(request: Request) -> None:
    """Raise 403 if the request carries an invalid API key."""
    key = request.headers.get("X-API-Key")
    if key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key.")


# ── helpers ────────────────────────────────────────────────────────────────────
def encode_frame(frame) -> bytes | None:
    """
    Encode a NumPy frame as JPEG bytes for WebSocket transmission.
    Returns None if encoding fails.
    Raw frame ~900KB → JPEG ~15KB at quality 75.
    """
    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality],
    )
    if not ok:
        return None
    return buffer.tobytes()


async def stream_camera(
    websocket: WebSocket,
    cam_key: str,
) -> None:
    await websocket.accept()
    logger.info(f"WebSocket client connected → {cam_key}")

    send_lock = asyncio.Lock()
    last_conn_state: bool | None = None
    last_enabled: bool | None = None

    async def safe_send_json(data: dict) -> None:
        async with send_lock:
            await websocket.send_json(data)

    async def safe_send_bytes(data: bytes) -> None:
        async with send_lock:
            await websocket.send_bytes(data)

    try:
        while True:
            is_enabled = cam_key in cameras

            # Notify the browser whenever enabled/disabled state changes
            if is_enabled != last_enabled:
                if is_enabled:
                    await safe_send_json({"type": "enabled", "camera": cam_key})
                else:
                    await safe_send_json({"type": "disabled", "camera": cam_key})
                last_enabled = is_enabled
                last_conn_state = None  # reset so reconnected fires when re-enabled

            if not is_enabled:
                await asyncio.sleep(1.0)
                continue

            # Re-fetch every iteration — cam could be toggled while we await
            camera = cameras[cam_key]
            queue = result_queues[cam_key]
            current_state = camera.is_connected

            if current_state != last_conn_state:
                if current_state:
                    await safe_send_json({"type": "reconnected", "camera": camera.name})
                    logger.info(f"[{cam_key}] Camera reconnected — notified browser.")
                else:
                    await safe_send_json({"type": "disconnected", "camera": camera.name})
                    logger.info(f"[{cam_key}] Camera disconnected — notified browser.")
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                last_conn_state = current_state

            if not current_state:
                await asyncio.sleep(0.5)
                continue

            try:
                result: DetectionResult = await asyncio.wait_for(
                    queue.get(),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                await safe_send_json({"type": "keepalive"})
                continue

            jpeg = encode_frame(result.frame)
            if jpeg:
                await safe_send_bytes(jpeg)

            await safe_send_json({
                "type": "meta",
                "camera": result.camera_name,
                "fps": round(result.fps, 1),
                "frame_number": result.frame_number,
                "detections": [
                    {
                        "track_id": d.track_id,
                        "class_name": d.class_name,
                        "confidence": round(d.confidence, 2),
                        "bbox": list(d.bbox),
                    }
                    for d in result.detections
                ],
                "metrics": get_system_metrics(),
                "counter": counters["cam0"].get_state() if "cam0" in counters else None,
            })

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected → {cam_key}")
    except Exception as e:
        logger.error(f"WebSocket error on {cam_key}: {e}")


# ── routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the live dashboard."""
    with open("static/dashboard.html", "r", encoding="utf-8") as f:
        return f.read()


@app.websocket("/ws/cam0")
async def ws_cam0(websocket: WebSocket):
    await stream_camera(websocket, "cam0")


@app.websocket("/ws/cam1")
async def ws_cam1(websocket: WebSocket):
    await stream_camera(websocket, "cam1")


@app.get("/api/status")
@limiter.limit("30/minute")
async def status(request: Request):
    """
    REST endpoint — current status of both cameras and detectors.
    Rate limited to 30 requests per minute.
    """
    return JSONResponse({
        "status": "running",
        "cameras": {
            key: {
                "name": cam.name,
                "connected": cam.is_connected,
                "fps": round(cam.fps, 1),
                "resolution": cam.get_resolution(),
            }
            for key, cam in cameras.items()
        },
        "detectors": {
            key: {
                "running": det.is_running,
                "fps": round(det.fps, 1),
            }
            for key, det in detectors.items()
        },
        "system": get_system_metrics(),
    })


@app.get("/api/health")
async def health():
    """Health check — used by Docker and monitoring tools."""
    return {"status": "ok", "timestamp": time.time()}


@app.post("/api/tripwire")
async def set_tripwire(request: Request):
    """
    Set the tripwire line coordinates for cam0.
    Body: { x1, y1, x2, y2, entry_direction }
    All coordinates normalized 0.0–1.0.
    """
    if "cam0" not in counters:
        raise HTTPException(status_code=503, detail="cam0 is not active")
    body = await request.json()
    counters["cam0"].set_tripwire(
        x1=float(body["x1"]),
        y1=float(body["y1"]),
        x2=float(body["x2"]),
        y2=float(body["y2"]),
        entry_direction=body.get("entry_direction", "positive"),
    )
    return {"status": "ok", "tripwire": body}


@app.post("/api/reset")
async def reset_counter(request: Request):
    """Reset the occupancy counter."""
    if "cam0" not in counters:
        raise HTTPException(status_code=503, detail="cam0 is not active")
    counters["cam0"].reset()
    return {"status": "ok"}


@app.post("/api/cameras/{cam_id}/enable")
async def enable_camera(cam_id: str):
    """Enable a camera at runtime — loads model if needed, starts stream and detector."""
    if cam_id not in ("cam0", "cam1"):
        raise HTTPException(status_code=400, detail="cam_id must be 'cam0' or 'cam1'")
    if cam_id in cameras:
        return {"status": "already_enabled", "camera": cam_id}
    idx = 0 if cam_id == "cam0" else 1
    await _start_cam(cam_id, idx)
    return {"status": "enabled", "camera": cam_id}


@app.post("/api/cameras/{cam_id}/disable")
async def disable_camera(cam_id: str):
    """Disable a camera at runtime — stops the stream and detector."""
    if cam_id not in ("cam0", "cam1"):
        raise HTTPException(status_code=400, detail="cam_id must be 'cam0' or 'cam1'")
    if cam_id not in cameras:
        return {"status": "already_disabled", "camera": cam_id}
    _stop_cam(cam_id)
    return {"status": "disabled", "camera": cam_id}