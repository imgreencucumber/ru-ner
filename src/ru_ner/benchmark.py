import platform
import time

import numpy as np


def hardware_info():
    import cpuinfo
    import onnxruntime
    import torch

    info = cpuinfo.get_cpu_info()
    return {
        "cpu": info["brand_raw"],
        "cpu_threads": info["count"],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
    }


def measure_latency(predict, sentences, warmup=20):
    """One sentence per call, like a single request to an API. Milliseconds per sentence."""
    for s in sentences[:warmup]:
        predict([s])
    times = []
    for s in sentences:
        start = time.perf_counter()
        predict([s])
        times.append(time.perf_counter() - start)
    ms = np.array(times) * 1000
    return {
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "mean_ms": float(ms.mean()),
    }


def measure_throughput(predict, sentences, warmup=64):
    """All sentences in one call, the model batches them itself. Sentences per second."""
    predict(sentences[:warmup])
    start = time.perf_counter()
    predict(sentences)
    return len(sentences) / (time.perf_counter() - start)
