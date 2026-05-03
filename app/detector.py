

import asyncio
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

from app.camera import StreamReader
from app.counter import LineCrossCounter

# Shared across all Detector instances — serializes GPU inference calls.
# Prevents CUDA context switching when multiple cameras run simultaneously.

@dataclass
class Detection:
    """A single detected and tracked object in one frame."""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class DetectionResult:
    """Full result for one processed frame from one camera."""
    camera_name: str
    frame: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    fps: float = 0.0
    frame_number: int = 0


class Detector:
    """
    Runs YOLO11 + ByteTrack in a background thread.

    Reads frames from a StreamReader, runs inference with persist=True
    so ByteTrack maintains track IDs across frames, then puts
    DetectionResult objects into an asyncio.Queue for the WebSocket handler.
    """

    def __init__(
        self,
        stream: StreamReader,
        model: YOLO,                  # ← was: model_path: str
        confidence: float = 0.40,
        device: str = "cpu",
        imgsz: int = 640,
        frame_skip: int = 1,
        result_queue: asyncio.Queue | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        counter: LineCrossCounter | None = None,
    ):
        self.stream = stream
        self.model_path = model.ckpt_path  # keep for logging
        self._model = model               # ← store directly, no loading needed
        self.confidence = confidence
        self.device = device
        self.imgsz = imgsz
        self.frame_skip = frame_skip
        self.result_queue = result_queue
        self.loop = loop
        self.counter = counter

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self.fps: float = 0.0
        self.frame_count: int = 0
        self.is_running: bool = False

    # ── public API ────────────────────────────────────────────────────

    def start(self) -> "Detector":
        """Start the inference thread. Model is already loaded externally."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._inference_loop,
            name=f"Detector-{self.stream.name}",
            daemon=True,
        )
        self._thread.start()
        self.is_running = True
        logger.info(f"[{self.stream.name}] Detector started.")
        return self

    def stop(self) -> None:
        """Stop the inference thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self.is_running = False
        logger.info(f"[{self.stream.name}] Detector stopped.")

    # ── internal ──────────────────────────────────────────────────────

    def _inference_loop(self) -> None:
        """
        Main inference loop.
        Reads frames from the StreamReader, runs YOLO + ByteTrack,
        and pushes results to the asyncio queue.
        """
        fps_timer = time.time()
        fps_frame_count = 0
        frame_number = 0

        while not self._stop_event.is_set():
            # Wait for the camera to connect before inferring
            if not self.stream.is_connected:
                time.sleep(0.1)
                continue

            ok, frame = self.stream.read()
            if not ok:
                time.sleep(0.005)
                continue

            frame_number += 1

            # Frame skip — reduces CPU/GPU load when set above 1
            if frame_number % self.frame_skip != 0:
                continue

            result = self._run_inference(frame, frame_number)

            # Update FPS counter
            fps_frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                self.fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_timer = time.time()

            result.fps = self.fps

            # Push result to the asyncio queue from this sync thread
            self._push_result(result)

    def _run_inference(self, frame: np.ndarray, frame_number: int) -> DetectionResult:

        results = self._model.track(
            source=frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
            device=self.device,
            classes=[0]
        )

        detections = []
        annotated_frame = frame.copy()
        frame_h, frame_w = frame.shape[:2]

        # Track which IDs are visible this frame
        # IDs not seen this frame get removed from counter history
        visible_ids: set[int] = set()

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                if box.id is None:
                    continue

                track_id  = int(box.id.item())
                class_id  = int(box.cls.item())
                confidence = float(box.conf.item())
                class_name = self._model.names[class_id]

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Centroid of the bounding box
                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)

                visible_ids.add(track_id)

                # Update the line cross counter if one is attached
                # Only count people — ignore chairs, vases, etc.
                if self.counter is not None and class_name == "person":
                    self.counter.update(
                        track_id=track_id,
                        cx=cx,
                        cy=cy,
                        frame_w=frame_w,
                        frame_h=frame_h,
                    )

                detections.append(Detection(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                ))

            annotated_frame = results[0].plot()

        # Remove disappeared track IDs from counter memory
        # so the same person can be counted again if they re-enter
        if self.counter is not None:
            prev_ids = set(self.counter._prev_positions.keys())
            for gone_id in prev_ids - visible_ids:
                self.counter.remove_track(gone_id)

        # Draw tripwire line and counter overlay on the annotated frame
        if self.counter is not None:
            self._draw_tripwire(annotated_frame, frame_w, frame_h)

        self._draw_overlay(annotated_frame)

        return DetectionResult(
            camera_name=self.stream.name,
            frame=annotated_frame,
            detections=detections,
            frame_number=frame_number,
        )

    def _draw_overlay(self, frame: np.ndarray) -> None:
        """Draw FPS counter and camera name onto the frame."""
        h, w = frame.shape[:2]

        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.putText(
            frame,
            f"{self.stream.name}  |  {self.fps:.1f} FPS",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_tripwire(
        self,
        frame: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> None:
        """
        Draw the tripwire line, entry/exit zone shading, direction arrows,
        and occupancy stats on the frame.

        Zone logic (horizontal line, entry_direction="positive"):
          - Below the line (higher Y) = ENTRY zone (green) — person is moving
            toward the camera, Y increases as they approach.
          - Above the line (lower Y)  = EXIT zone  (red)  — person is moving
            away from the camera, Y decreases as they recede.
        Swapped automatically when entry_direction="negative".
        """
        state = self.counter.get_state()
        tw    = state["tripwire"]

        x1 = int(tw["x1"] * frame_w)
        y1 = int(tw["y1"] * frame_h)
        x2 = int(tw["x2"] * frame_w)
        y2 = int(tw["y2"] * frame_h)

        entry_dir = tw["entry_direction"]  # "positive" or "negative"

        # ── Zone shading ──────────────────────────────────────────────
        # Two polygons that tile the frame along the tripwire line.
        # "below" = the polygon from the line down to the bottom edge.
        # "above" = the polygon from the line up to the top edge.
        below_pts = np.array(
            [[x1, y1], [x2, y2], [frame_w, frame_h], [0, frame_h]], np.int32
        )
        above_pts = np.array(
            [[x1, y1], [x2, y2], [frame_w, 0], [0, 0]], np.int32
        )

        if entry_dir == "positive":
            # crossing into below = ENTRY (Y increases = approaching camera)
            entry_poly, exit_poly = below_pts, above_pts
        else:
            entry_poly, exit_poly = above_pts, below_pts

        overlay = frame.copy()
        cv2.fillPoly(overlay, [entry_poly], (0, 160, 0))   # green = entry
        cv2.fillPoly(overlay, [exit_poly],  (0, 0, 160))   # red   = exit
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        # ── Zone labels ───────────────────────────────────────────────
        y_mid = (y1 + y2) // 2

        if entry_dir == "positive":
            entry_label_y = min(frame_h - 12, (y_mid + frame_h) // 2)
            exit_label_y  = max(20,            y_mid // 2)
        else:
            entry_label_y = max(20,            y_mid // 2)
            exit_label_y  = min(frame_h - 12, (y_mid + frame_h) // 2)

        for text, pos_y, color in [
            ("ENTRY", entry_label_y, (0, 230, 0)),
            ("EXIT",  exit_label_y,  (0, 60, 230)),
        ]:
            (tw_w, tw_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            )
            lx = (frame_w - tw_w) // 2
            cv2.rectangle(
                frame,
                (lx - 4, pos_y - tw_h - 4),
                (lx + tw_w + 4, pos_y + 4),
                (0, 0, 0), -1,
            )
            cv2.putText(
                frame, text, (lx, pos_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
            )

        # ── Direction arrows on the line ──────────────────────────────
        # Green arrow points into the entry zone; red arrow into exit zone.
        arrow_x = frame_w // 2
        arrow_len = 22

        if entry_dir == "positive":
            entry_tip_y = y_mid + arrow_len   # downward = toward camera
            exit_tip_y  = y_mid - arrow_len   # upward   = away from camera
        else:
            entry_tip_y = y_mid - arrow_len
            exit_tip_y  = y_mid + arrow_len

        cv2.arrowedLine(
            frame,
            (arrow_x - 18, y_mid), (arrow_x - 18, entry_tip_y),
            (0, 230, 0), 2, cv2.LINE_AA, tipLength=0.45,
        )
        cv2.arrowedLine(
            frame,
            (arrow_x + 18, y_mid), (arrow_x + 18, exit_tip_y),
            (0, 60, 230), 2, cv2.LINE_AA, tipLength=0.45,
        )

        # ── Tripwire line ─────────────────────────────────────────────
        line_color = (0, 0, 255) if state["capacity_exceeded"] else (0, 255, 100)
        cv2.line(frame, (x1, y1), (x2, y2), line_color, 3, cv2.LINE_AA)
        cv2.circle(frame, (x1, y1), 6, line_color, -1)
        cv2.circle(frame, (x2, y2), 6, line_color, -1)

        # ── Counter label at line midpoint ────────────────────────────
        mx = (x1 + x2) // 2
        label = (
            f"IN:{state['entries']}  "
            f"OUT:{state['exits']}  "
            f"OCC:{state['occupancy']}/{state['max_occupancy']}"
        )
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(
            frame,
            (mx - 4, y_mid - lh - 6),
            (mx + lw + 4, y_mid + 4),
            (0, 0, 0), -1,
        )
        cv2.putText(
            frame, label, (mx, y_mid),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, line_color, 1, cv2.LINE_AA,
        )

    def _push_result(self, result: DetectionResult) -> None:
        """
        Thread-safe push to asyncio.Queue.
        If the queue is full, drop the oldest result to stay current.
        Uses call_soon_threadsafe to safely communicate between
        the sync inference thread and the async event loop.
        """
        if self.result_queue is None or self.loop is None:
            return

        async def _put():
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await self.result_queue.put(result)

        asyncio.run_coroutine_threadsafe(_put(), self.loop)