import cv2
import time
import asyncio
from camera import StreamReader
from detector import Detector
from config import settings

async def main():
    # One result queue per camera
    queue0 = asyncio.Queue(maxsize=2)
    queue1 = asyncio.Queue(maxsize=2)
    loop = asyncio.get_running_loop()

    cam0 = StreamReader(
        source=settings.get_camera_source(0),
        name=settings.camera_0_name,
        width=settings.camera_width,
        height=settings.camera_height,
        backend=settings.camera_backend,
    ).start()

    cam1 = StreamReader(
        source=settings.get_camera_source(1),
        name=settings.camera_1_name,
        width=settings.camera_width,
        height=settings.camera_height,
        backend=settings.camera_backend,
    ).start()

    await asyncio.sleep(2)  # let cameras connect

    det0 = Detector(
        stream=cam0,
        model_path=settings.model_path,
        confidence=settings.confidence_threshold,
        device=settings.device,
        imgsz=settings.inference_size,
        frame_skip=settings.frame_skip,
        result_queue=queue0,
        loop=loop,
    ).start()

    det1 = Detector(
        stream=cam1,
        model_path=settings.model_path,
        confidence=settings.confidence_threshold,
        device=settings.device,
        imgsz=settings.inference_size,
        frame_skip=settings.frame_skip,
        result_queue=queue1,
        loop=loop,
    ).start()

    print("Running for 15 seconds — watch for detections...")
    start = time.time()

    while time.time() - start < 50:
        for q, name in [(queue0, settings.camera_0_name), (queue1, settings.camera_1_name)]:
            try:
                result = q.get_nowait()
                cv2.imshow(name, result.frame)
                if result.detections:
                    for d in result.detections:
                        print(f"[{name}] ID:{d.track_id} {d.class_name} {d.confidence:.2f}")
            except asyncio.QueueEmpty:
                pass

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        await asyncio.sleep(0.01)

    det0.stop()
    det1.stop()
    cam0.stop()
    cam1.stop()
    cv2.destroyAllWindows()
    print("Phase 3 test complete.")

asyncio.run(main())