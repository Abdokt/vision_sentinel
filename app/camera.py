import cv2
import threading
import queue
import time
from loguru import logger


class StreamReader:
    """
    Reads frames from a webcam or RTSP stream in a background thread.
    Frames are placed in a bounded queue (maxsize=2) so the consumer
    always gets the latest frame, never a stale backlog.
    """

    def __init__(
        self,
        source: int | str,
        name: str = "Camera",
        queue_size: int = 2,
        reconnect_delay: float = 3.0,
        width: int = 640,
        height: int = 480,
        backend: str = "dshow",
    ):
        self.source = source
        self.name = name
        self.queue_size = queue_size
        self.reconnect_delay = reconnect_delay
        self.width = width
        self.height = height
        self.backend = backend

        self._frame_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stop_reading = threading.Event()  # set externally to interrupt _read_frames
        self._reconnect_event = threading.Event()  # set by watchdog when device reappears

        self.fps: float = 0.0
        self.frame_count: int = 0
        self.is_connected: bool = False

    # ── public API ────────────────────────────────────────────────────

    def start(self) -> "StreamReader":
        """Start the background reader thread. Returns self for chaining."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"StreamReader-{self.name}",
            daemon=True,  # thread dies automatically when main process exits
        )
        self._thread.start()
        logger.info(f"[{self.name}] Stream reader started — source: {self._safe_source()}")
        return self

    def stop(self) -> None:
        """Signal the reader thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._cap:
            self._cap.release()
        logger.info(f"[{self.name}] Stream reader stopped.")

    def read(self) -> tuple[bool, cv2.typing.MatLike | None]:
        """
        Get the latest frame from the queue.
        Returns (True, frame) if a frame is available, (False, None) otherwise.
        Non-blocking — never hangs the caller.
        """
        try:
            frame = self._frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            return False, None

    def force_disconnect(self) -> None:
        """
        Called by the Windows device watchdog when the physical camera
        disappears from the OS device list. Sets is_connected=False and
        signals _read_frames to exit immediately — bypasses DirectShow's
        failure to report USB disconnection through cap.read().
        """
        if self.is_connected:
            self.is_connected = False
            self._stop_reading.set()
            logger.warning(f"[{self.name}] Force-disconnected by device watchdog.")

    def request_reconnect(self) -> None:
        """
        Called by the watchdog when the physical device reappears in the OS.
        Signals the reader loop to attempt reconnection.
        """
        logger.info(f"[{self.name}] Reconnect requested by watchdog.")
        self._reconnect_event.set()

    def get_resolution(self) -> tuple[int, int]:
        """Returns (width, height) of the stream, or (0, 0) if not connected."""
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 0, 0

    # ── internal ──────────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        """
        Main loop running in the background thread.

        Behaviour differs by source type:
        - RTSP URL (str): auto-reconnects indefinitely — network cameras drop and recover.
        - Webcam index (int): connects once. If it fails or drops, marks as
          permanently disconnected and stops retrying. Reconnection only happens
          when start() is called again externally.
          This prevents Windows from rebinding a different physical camera to
          the same index after the original is unplugged.
        """
        is_rtsp = isinstance(self.source, str) and self.source.startswith("rtsp://")

        self._connect()

        if not self.is_connected:
            if is_rtsp:
                # RTSP — keep retrying
                while not self._stop_event.is_set() and not self.is_connected:
                    logger.warning(
                        f"[{self.name}] Could not connect. "
                        f"Retrying in {self.reconnect_delay}s..."
                    )
                    time.sleep(self.reconnect_delay)
                    self._connect()
            else:
                logger.error(
                    f"[{self.name}] Could not open webcam index {self.source}. "
                    f"Check the camera is plugged in and not used by another app."
                )
                return

        # Connected — read frames
        while not self._stop_event.is_set():
            self._read_frames()

            if self._stop_event.is_set():
                break

            # Stream dropped
            self.is_connected = False
            if self._cap:
                self._cap.release()

            if is_rtsp:
                logger.warning(f"[{self.name}] Stream lost. Reconnecting in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)
                self._connect()
            else:
                # Webcam physically disconnected.
                # Wait for the watchdog to signal the device is back
                # before attempting reconnection — this prevents binding
                # to the wrong camera index on Windows.
                logger.warning(
                    f"[{self.name}] Webcam disconnected. "
                    f"Waiting for device to reappear..."
                )
                self._reconnect_event.clear()

                while not self._stop_event.is_set():
                    signaled = self._reconnect_event.wait(timeout=1.0)
                    if signaled:
                        self._reconnect_event.clear()
                        logger.info(f"[{self.name}] Attempting reconnection...")
                        self._connect()
                        if self.is_connected:
                            break  # exit inner wait loop, outer loop calls _read_frames
                        else:
                            logger.warning(
                                f"[{self.name}] Reconnect failed. "
                                f"Waiting for next signal..."
                            )

    def _connect(self) -> None:
        try:
            # Use DirectShow on Windows for reliable multi-camera support
            backend_id = cv2.CAP_DSHOW if self.backend == "dshow" else cv2.CAP_ANY
            self._cap = cv2.VideoCapture(self.source, backend_id)

            if not self._cap.isOpened():
                self.is_connected = False
                return

            # Explicitly set resolution — prevents cameras defaulting to max res
            # which causes USB bandwidth starvation when two cameras run together
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Read back actual values — camera may not support the exact requested size
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cam_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30

            logger.success(
                f"[{self.name}] Connected — {actual_w}x{actual_h} @ {cam_fps:.0f}fps"
            )
            self.is_connected = True

        except Exception as e:
            logger.error(f"[{self.name}] Connection error: {e}")
            self.is_connected = False

    def _read_frames(self) -> None:
        """
        Read frames in a tight loop until the stream fails or stop is requested.
        Validates every frame before queuing — discards malformed frames that
        occur due to USB buffer hiccups on Windows (shows as tiled mosaic).
        """
        fps_timer = time.time()
        fps_frame_count = 0

        # Lock in the expected dimensions at the start of this session
        expected_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        expected_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        consecutive_failures = 0
        max_consecutive_failures = 10  # reconnect after this many bad frames

        # Clear the stop signal at the start of each new read session
        self._stop_reading.clear()

        while not self._stop_event.is_set() and not self._stop_reading.is_set():
            if not self._cap or not self._cap.isOpened():
                break

            ret, frame = self._cap.read()

            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(
                        f"[{self.name}] {consecutive_failures} consecutive "
                        f"read failures — reconnecting."
                    )
                    break
                time.sleep(0.005)
                continue

            # Validate frame dimensions — malformed USB frames have wrong shape
            actual_h, actual_w = frame.shape[:2]
            if actual_w != expected_w or actual_h != expected_h:
                logger.debug(
                    f"[{self.name}] Malformed frame discarded — "
                    f"expected {expected_w}x{expected_h}, "
                    f"got {actual_w}x{actual_h}"
                )
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(f"[{self.name}] Too many malformed frames — reconnecting.")
                    break
                continue

            # Frame is valid — reset failure counter
            consecutive_failures = 0

            # Drop oldest frame if queue is full, then add the new one
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass

            self._frame_queue.put(frame)
            self.frame_count += 1
            fps_frame_count += 1

            # Update FPS every second
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                self.fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_timer = time.time()

    def _safe_source(self) -> str:
        """
        Returns a log-safe version of the source.
        Masks credentials in RTSP URLs so they never appear in logs.
        e.g. rtsp://admin:secret@192.168.1.1/... → rtsp://***@192.168.1.1/...
        """
        src = str(self.source)
        if src.startswith("rtsp://") and "@" in src:
            protocol, rest = src.split("://", 1)
            credentials, address = rest.split("@", 1)
            return f"{protocol}://***@{address}"
        return src