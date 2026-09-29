"""人员库：PostgreSQL 存 teamusers 用户 id 与 512 维人脸特征，余弦相似度检索。

设计要点：
- 只存 teamusers 规范用户 id 与特征向量，不存姓名等任何其他内容；
- 特征向量 L2 归一化后以 BYTEA 存储；
- 检索走进程内缓存矩阵：增删后失效，下次检索时重建。
  所有向量已归一化，矩阵点积即余弦相似度，万级规模单次检索 <10ms；
- 单连接 + 一把锁：所有读写互斥，避免跨线程使用同一连接的竞态。
"""
import threading

import numpy as np
import psycopg
import psycopg.conninfo

SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
    user_id     TEXT        PRIMARY KEY,
    embedding   BYTEA       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def ensure_database(dsn: str) -> None:
    """目标数据库不存在时自动创建（借 postgres 维护库执行 CREATE DATABASE）。

    不依赖错误文案判定"库不存在"（错误信息随系统语言变化），
    直接查 pg_database。
    """
    params = psycopg.conninfo.conninfo_to_dict(dsn)
    dbname = params.pop("dbname", None) or "face_recognition"
    params["dbname"] = "postgres"
    admin_dsn = psycopg.conninfo.make_conninfo(**params)
    with psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone()
        if not exists:
            # 标识符无法参数化，库名来自配置而非用户输入
            conn.execute(f'CREATE DATABASE "{dbname}"')


class FaceStore:
    def __init__(self, dsn: str) -> None:
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(SCHEMA)
        self._cache_ids: list[str] = []
        self._cache_matrix: np.ndarray | None = None

    def _invalidate_cache(self) -> None:
        self._cache_ids = []
        self._cache_matrix = None

    def add(self, user_id: str, embedding: np.ndarray) -> None:
        """登记人脸；同一 user_id 重复登记时覆盖旧特征。"""
        with self._lock:
            self._conn.execute(
                "INSERT INTO faces (user_id, embedding) VALUES (%s, %s) "
                "ON CONFLICT (user_id) DO UPDATE SET embedding = EXCLUDED.embedding",
                (user_id, embedding.astype(np.float32).tobytes()),
            )
            self._invalidate_cache()

    def exists(self, user_id: str) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM faces WHERE user_id = %s", (user_id,)).fetchone()
        return row is not None

    def list(self) -> list[str]:
        """返回所有已登记的 user_id。"""
        with self._lock:
            rows = self._conn.execute("SELECT user_id FROM faces ORDER BY user_id").fetchall()
        return [r[0] for r in rows]

    def delete(self, user_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM faces WHERE user_id = %s", (user_id,))
            self._invalidate_cache()
            return cur.rowcount > 0

    def search(self, embedding: np.ndarray, threshold: float) -> tuple[str | None, float | None]:
        """返回 (最相似的 user_id, 相似度)。

        相似度未达阈值时 user_id 为 None，但仍返回最佳相似度便于排查；
        库为空时返回 (None, None)。
        """
        with self._lock:
            if self._cache_matrix is None:
                rows = self._conn.execute("SELECT user_id, embedding FROM faces").fetchall()
                self._cache_ids = [r[0] for r in rows]
                self._cache_matrix = (
                    np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows]) if rows else None
                )
            ids, matrix = self._cache_ids, self._cache_matrix

            if matrix is None:
                return None, None

            sims = matrix @ embedding  # 双方均已归一化 → 点积 = 余弦相似度
            best = int(np.argmax(sims))
            best_sim = round(float(sims[best]), 4)

        if best_sim < threshold:
            return None, best_sim
        return ids[best], best_sim

    def close(self) -> None:
        with self._lock:
            self._conn.close()
