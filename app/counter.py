import threading
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class TripwireConfig:
    """
    A tripwire line defined by two points in pixel coordinates.
    x1,y1 = start point, x2,y2 = end point.
    entry_direction: "positive" means crossing from left-to-right
    or top-to-bottom relative to the line normal is an entry.
    """
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 0.5
    entry_direction: str = "positive"  # "positive" or "negative"


@dataclass
class CrossingEvent:
    track_id: int
    direction: str   # "entry" or "exit"
    position: tuple[float, float]


class LineCrossCounter:
    """
    Counts people crossing a virtual tripwire line using ByteTrack IDs.

    How crossing detection works:
    - Every frame we receive the centroid of each tracked person.
    - We compare their current centroid to their previous centroid.
    - We compute which side of the tripwire line each point is on
      using the cross product (sign of the scalar).
    - If the sign flipped between frames, they crossed the line.
    - The sign of the cross product also tells us direction —
      entry vs exit.
    - We remember which IDs have already crossed so the same
      physical crossing is never counted twice.
    """

    def __init__(self, camera_name: str, max_occupancy: int = 3):
        self.camera_name = camera_name
        self.max_occupancy = max_occupancy

        self._lock = threading.Lock()
        self._tripwire = TripwireConfig()

        # Occupancy counters
        self.entries: int = 0
        self.exits: int = 0

        # Track history — maps track_id → last known centroid
        self._prev_positions: dict[int, tuple[float, float]] = {}

        # IDs that have already triggered a crossing this session
        # prevents double-counting if someone lingers on the line
        self._crossed_ids: set[int] = set()

        # Recent crossing events for the dashboard log
        self._recent_events: list[CrossingEvent] = []

        logger.info(f"[Counter:{camera_name}] Initialized. Max occupancy: {max_occupancy}")

    # ── public API ────────────────────────────────────────────────────

    @property
    def occupancy(self) -> int:
        """Current occupancy = entries - exits, floored at 0."""
        return max(0, self.entries - self.exits)

    @property
    def capacity_exceeded(self) -> bool:
        return self.occupancy >= self.max_occupancy

    def set_tripwire(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        entry_direction: str = "positive",
    ) -> None:
        """
        Set the tripwire line coordinates.
        Coordinates are normalized 0.0–1.0 relative to frame size
        so they work regardless of resolution.
        """
        with self._lock:
            self._tripwire = TripwireConfig(x1, y1, x2, y2, entry_direction)
            # Reset crossing memory when line moves — prevents ghost counts
            self._crossed_ids.clear()
            self._prev_positions.clear()
        logger.info(
            f"[Counter:{self.camera_name}] Tripwire set: "
            f"({x1:.2f},{y1:.2f}) → ({x2:.2f},{y2:.2f})"
        )

    def reset(self) -> None:
        """Reset all counters and history."""
        with self._lock:
            self.entries = 0
            self.exits = 0
            self._prev_positions.clear()
            self._crossed_ids.clear()
            self._recent_events.clear()
        logger.info(f"[Counter:{self.camera_name}] Counters reset.")

    def update(
        self,
        track_id: int,
        cx: float,
        cy: float,
        frame_w: int,
        frame_h: int,
    ) -> CrossingEvent | None:
        """
        Update position for one tracked person and check for line crossing.
        cx, cy are pixel coordinates — we normalize internally.
        Returns a CrossingEvent if a crossing was detected, else None.
        """
        # Normalize to 0.0–1.0
        nx = cx / frame_w
        ny = cy / frame_h

        with self._lock:
            tw = self._tripwire
            prev = self._prev_positions.get(track_id)
            self._prev_positions[track_id] = (nx, ny)

            if prev is None:
                # First time we see this ID — record position, no crossing possible
                return None

            if track_id in self._crossed_ids:
                # This ID already triggered a crossing — ignore until it disappears
                return None

            prev_side = self._side(prev[0], prev[1], tw)
            curr_side = self._side(nx, ny, tw)

            if prev_side == 0 or curr_side == 0:
                # Exactly on the line — skip to avoid ambiguity
                return None

            if prev_side != curr_side:
                # Crossing detected
                self._crossed_ids.add(track_id)

                if curr_side > 0:
                    is_entry = tw.entry_direction == "positive"
                else:
                    is_entry = tw.entry_direction == "negative"

                direction = "entry" if is_entry else "exit"

                if is_entry:
                    self.entries += 1
                else:
                    self.exits = min(self.exits + 1, self.entries)

                event = CrossingEvent(
                    track_id=track_id,
                    direction=direction,
                    position=(nx, ny),
                )
                self._recent_events.append(event)
                if len(self._recent_events) > 50:
                    self._recent_events.pop(0)

                logger.info(
                    f"[Counter:{self.camera_name}] "
                    f"ID:{track_id} → {direction.upper()} | "
                    f"occupancy: {self.occupancy}"
                )
                return event

        return None

    def remove_track(self, track_id: int) -> None:
        """
        Call when a track ID disappears from the frame.
        Removes from crossed_ids so if the person re-enters
        they get counted again as a new visit.
        """
        with self._lock:
            self._prev_positions.pop(track_id, None)
            self._crossed_ids.discard(track_id)

    def get_state(self) -> dict:
        """Return full counter state as a serializable dict."""
        with self._lock:
            tw = self._tripwire
            return {
                "entries": self.entries,
                "exits": self.exits,
                "occupancy": self.occupancy,
                "capacity_exceeded": self.capacity_exceeded,
                "max_occupancy": self.max_occupancy,
                "tripwire": {
                    "x1": tw.x1, "y1": tw.y1,
                    "x2": tw.x2, "y2": tw.y2,
                    "entry_direction": tw.entry_direction,
                },
                "recent_events": [
                    {
                        "track_id": e.track_id,
                        "direction": e.direction,
                    }
                    for e in self._recent_events[-10:]
                ],
            }

    # ── geometry ──────────────────────────────────────────────────────

    def _side(
        self,
        px: float, py: float,
        tw: TripwireConfig,
    ) -> int:
        """
        Returns which side of the tripwire line a point is on.
        Uses the sign of the 2D cross product of the line vector
        and the vector from the line start to the point.

        Returns:
          +1 if point is on the positive side
          -1 if point is on the negative side
           0 if point is exactly on the line
        """
        # Line vector
        dx = tw.x2 - tw.x1
        dy = tw.y2 - tw.y1

        # Vector from line start to point
        px_rel = px - tw.x1
        py_rel = py - tw.y1

        # 2D cross product (scalar)
        cross = dx * py_rel - dy * px_rel

        if cross > 1e-9:
            return 1
        elif cross < -1e-9:
            return -1
        return 0