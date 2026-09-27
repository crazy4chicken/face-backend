"""人员库：PostgreSQL 存人员信息与 512 维人脸特征，余弦相似度检索。

设计要点：
- 特征向量 L2 归一化后以 BYTEA 存储，info 用 JSONB 存任意扩展字段；
- 检索走进程内缓存矩阵：增删后失效，下次检索时重建。
  所有向量已归一化，矩阵点积即余弦相似度，万级规模单次检索 <10ms；
- 单连接 + 一把锁：所有读写互斥，避免跨线程使用同一连接的竞态。
"""
import json
import threading

import numpy as np
import psycopg
import psycopg.conninfo

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id          SERIAL PRIMARY KEY,
    name        TEXT        NOT NULL,
    info        JSONB       NOT NULL DEFAULT '{}',
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


class PersonStore:
    def __init__(self, dsn: str) -> None:
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(SCHEMA)
        self._cache_rows: list[dict] = []
        self._cache_matrix: np.ndarray | None = None

    @staticmethod
    def _row_to_dict(row) -> dict:
        return {
            "id": row[0],
            "name": row[1],
            "info": row[2] if isinstance(row[2], dict) else json.loads(row[2]),
            "created_at": row[3].isoformat(),
        }

    def _invalidate_cache(self) -> None:
        self._cache_rows = []
        self._cache_matrix = None

    def add(self, name: str, info: dict, embedding: np.ndarray) -> int:
        with self._lock:
            row = self._conn.execute(
                "INSERT INTO persons (name, info, embedding) VALUES (%s, %s, %s) RETURNING id",
                (name, json.dumps(info, ensure_ascii=False), embedding.astype(np.float32).tobytes()),
            ).fetchone()
            self._invalidate_cache()
            return row[0]

    def get(self, person_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, info, created_at FROM persons WHERE id = %s", (person_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT id, name, info, created_at FROM persons ORDER BY id").fetchall()
        return [self._row_to_dict(r) for r in rows]

    def delete(self, person_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM persons WHERE id = %s", (person_id,))
            self._invalidate_cache()
            return cur.rowcount > 0

    def search(self, embedding: np.ndarray, threshold: float) -> tuple[dict | None, float | None]:
        """返回 (最相似人员, 相似度)。

        相似度未达阈值时人员为 None，但仍返回最佳相似度便于排查；
        人员库为空时返回 (None, None)。
        """
        with self._lock:
            if self._cache_matrix is None:
                rows = self._conn.execute(
                    "SELECT id, name, info, created_at, embedding FROM persons"
                ).fetchall()
                self._cache_rows = rows
                self._cache_matrix = (
                    np.stack([np.frombuffer(r[4], dtype=np.float32) for r in rows]) if rows else None
                )
            rows, matrix = self._cache_rows, self._cache_matrix

            if matrix is None:
                return None, None

            sims = matrix @ embedding  # 双方均已归一化 → 点积 = 余弦相似度
            best = int(np.argmax(sims))
            best_sim = round(float(sims[best]), 4)

        if best_sim < threshold:
            return None, best_sim
        return self._row_to_dict(rows[best]), best_sim

    def close(self) -> None:
        with self._lock:
            self._conn.close()
