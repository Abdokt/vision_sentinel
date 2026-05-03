import psutil
from loguru import logger

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def get_system_metrics() -> dict:
    """
    Returns current CPU, RAM, and GPU metrics as a dictionary.
    Gracefully handles machines with no NVIDIA GPU.
    """
    metrics = {
        "cpu_percent": round(psutil.cpu_percent(interval=None), 1),
        "ram_percent": round(psutil.virtual_memory().percent, 1),
        "ram_used_gb": round(psutil.virtual_memory().used / 1e9, 2),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "gpu_available": False,
        "gpu_percent": 0.0,
        "gpu_mem_used_mb": 0,
        "gpu_mem_total_mb": 0,
        "gpu_name": "N/A",
    }

    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # use first GPU
                metrics.update({
                    "gpu_available": True,
                    "gpu_percent": round(gpu.load * 100, 1),
                    "gpu_mem_used_mb": round(gpu.memoryUsed),
                    "gpu_mem_total_mb": round(gpu.memoryTotal),
                    "gpu_name": gpu.name,
                })
        except Exception as e:
            logger.debug(f"GPU metrics error: {e}")

    return metrics