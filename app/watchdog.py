import subprocess
import threading

from loguru import logger

from app.camera import StreamReader


class WindowsDeviceWatchdog:
    """
    Monitors Windows PnP devices for camera disconnection.

    DirectShow (used by OpenCV on Windows) does not reliably signal when
    a USB camera is physically unplugged — cap.read() silently rebinds
    to a different camera instead of returning False.

    This watchdog bypasses DirectShow by querying the Windows PnP device
    list via PowerShell every N seconds. When a monitored device disappears
    from the OK device list, it calls force_disconnect() on the StreamReader
    so the rest of the pipeline reacts immediately and correctly.
    """

    def __init__(self, check_interval: float = 2.0):
        self.check_interval = check_interval
        self._watches: list[dict] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def watch(
        self,
        device_name: str,
        stream: StreamReader,
    ) -> "WindowsDeviceWatchdog":
        """
        Register a device to monitor, or update its stream reference if already
        registered (safe to call again after a runtime enable/disable cycle).
        device_name must match the FriendlyName from Get-PnpDevice exactly.
        """
        for entry in self._watches:
            if entry["device_name"] == device_name:
                entry["stream"] = stream
                entry["last_known_present"] = None  # re-check on next tick
                logger.info(f"[Watchdog] Updated stream for: '{device_name}'")
                return self
        self._watches.append({
            "device_name": device_name,
            "stream": stream,
            "last_known_present": None,
        })
        logger.info(f"[Watchdog] Monitoring device: '{device_name}'")
        return self

    def start(self) -> "WindowsDeviceWatchdog":
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._watch_loop,
            name="DeviceWatchdog",
            daemon=True,
        )
        self._thread.start()
        logger.info("[Watchdog] Windows device watchdog started.")
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("[Watchdog] Stopped.")

    # ── internal ──────────────────────────────────────────────────────

    def _get_ok_device_names(self) -> set[str]:
        """
        Query Windows for all PnP devices currently in OK status.
        Returns a set of FriendlyName strings.
        Times out after 5 seconds to avoid blocking the watchdog thread.
        """
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "Get-PnpDevice -Status OK | Select-Object -ExpandProperty FriendlyName",
                ],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if result.returncode == 0:
                return {
                    line.strip()
                    for line in result.stdout.splitlines()
                    if line.strip()
                }
        except subprocess.TimeoutExpired:
            logger.debug("[Watchdog] PowerShell query timed out.")
        except Exception as e:
            logger.debug(f"[Watchdog] Query error: {e}")
        return set()

    def _watch_loop(self) -> None:
        while not self._stop_event.is_set():
            ok_devices = self._get_ok_device_names()

            for watch in self._watches:
                device_name: str = watch["device_name"]
                stream: StreamReader = watch["stream"]
                is_present: bool = device_name in ok_devices

                # First check — just record the initial state, don't act
                if watch["last_known_present"] is None:
                    watch["last_known_present"] = is_present
                    logger.info(
                        f"[Watchdog] '{device_name}' initial state: "
                        f"{'present' if is_present else 'NOT present'}"
                    )
                    continue

                # State changed
                if is_present != watch["last_known_present"]:
                    if not is_present:
                        # Device disappeared — force disconnect immediately
                        logger.warning(
                            f"[Watchdog] '{device_name}' disappeared from OS. "
                            f"Forcing disconnect."
                        )
                        stream.force_disconnect()
                    else:
                        # Device reappeared — signal the StreamReader to reconnect
                        logger.success(
                            f"[Watchdog] '{device_name}' reappeared. "
                            f"Signaling reconnect."
                        )
                        stream.request_reconnect()
                    watch["last_known_present"] = is_present

            # Wait for next check interval
            self._stop_event.wait(self.check_interval)
