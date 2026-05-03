from app.counter import LineCrossCounter

c = LineCrossCounter("TestCam", max_occupancy=3)

# Set a horizontal line across the middle of a 640x480 frame
# normalized: y=0.5, full width
c.set_tripwire(0.0, 0.5, 1.0, 0.5, entry_direction="positive")

# Simulate person walking top to bottom (entering)
# frame is 640x480
# track_id=1 starts at y=100 (above line), moves to y=350 (below line)
e1 = c.update(1, cx=320, cy=100, frame_w=640, frame_h=480)
print(f"Frame 1: {e1}")  # should be None — first sighting

e2 = c.update(1, cx=320, cy=350, frame_w=640, frame_h=480)
print(f"Frame 2: {e2}")  # should be CrossingEvent direction=entry

print(f"Occupancy: {c.occupancy}")  # should be 1
print(f"Entries: {c.entries}, Exits: {c.exits}")

# Simulate person walking bottom to top (exiting)
e3 = c.update(2, cx=320, cy=400, frame_w=640, frame_h=480)
e4 = c.update(2, cx=320, cy=200, frame_w=640, frame_h=480)
print(f"Frame 3: {e3}")  # None
print(f"Frame 4: {e4}")  # CrossingEvent direction=exit

print(f"Occupancy: {c.occupancy}")  # should be 0
print(f"Entries: {c.entries}, Exits: {c.exits}")

# Simulate capacity breach
c.update(3, cx=320, cy=100, frame_w=640, frame_h=480)
c.update(3, cx=320, cy=350, frame_w=640, frame_h=480)
c.update(4, cx=320, cy=100, frame_w=640, frame_h=480)
c.update(4, cx=320, cy=350, frame_w=640, frame_h=480)
c.update(5, cx=320, cy=100, frame_w=640, frame_h=480)
c.update(5, cx=320, cy=350, frame_w=640, frame_h=480)
print(f"Occupancy: {c.occupancy}")  # should be 3
print(f"Capacity exceeded: {c.capacity_exceeded}")  # should be True

print("Counter test complete.")