"""InsightFace 推理引擎封装。

FaceAnalysis 不是线程安全的，所有推理经同一把锁串行化。
FastAPI 的同步路由默认跑在线程池里，这把锁保证并发请求不互相踩踏。
"""
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from . import config


@dataclass
class DetectedFace:
    bbox: list[float]          # [x1, y1, x2, y2]
    det_score: float           # 检测置信度
    embedding: np.ndarray      # 512 维 L2 归一化特征


class FaceEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        app = FaceAnalysis(name=config.MODEL_PACK, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=config.DET_SIZE)
        self._app = app

    def analyze(self, image_bytes: bytes) -> list[DetectedFace]:
        """解码图片并返回所有人脸（含特征向量）。图片非法时抛 ValueError。"""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解码图片，请上传 JPG/PNG 等常见格式")

        # 防像素炸弹：压缩文件可以很小，解码后像素却可能高达数亿
        if img.shape[0] * img.shape[1] > config.MAX_PIXELS:
            raise ValueError(f"图片像素超过上限 {config.MAX_PIXELS}")

        # SCRFD 对过小图片（<300px）检出率差：先放大到短边 384px
        h, w = img.shape[:2]
        short = min(h, w)
        inv_scale = 1.0
        if short < 384:
            scale = 384.0 / short
            inv_scale = 1.0 / scale
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)

        with self._lock:
            faces = self._app.get(img)

        results = [
            DetectedFace(
                bbox=[round(float(v) * inv_scale, 2) for v in f.bbox],
                det_score=round(float(f.det_score), 4),
                embedding=f.normed_embedding.astype(np.float32),
            )
            for f in faces
        ]
        # 按人脸面积从大到小排序，并限制数量
        results.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return results[: config.MAX_FACES]


_engine: FaceEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> FaceEngine:
    """懒加载单例：模型首次使用时才下载/加载。双重检查锁防止并发首请求重复建引擎。"""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = FaceEngine()
    return _engine
