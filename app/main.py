"""人脸识别后端 API。

约定：注册照片必须为单人照；识别照片可含多张人脸（通常不多）。
人员以 teamusers 用户 id 标识，本服务不存姓名等任何其他信息。

鉴权：所有业务接口要求 teamusers 访问令牌；
识别/查询需要 face:check:any，注册/删除需要 face:modify:any。

路由：
    POST   /faces/{user_id}   登记人脸（单人照片；多张人脸时拒绝）
    GET    /faces             列出所有已登记的 user_id
    GET    /faces/{user_id}   查询某人是否已登记
    DELETE /faces/{user_id}   删除登记
    POST   /recognize         识别照片中的所有人脸，逐脸返回匹配的 user_id
    GET    /healthz           健康检查（无需鉴权，供探活）

启动：uv run uvicorn app.main:app --host 0.0.0.0 --port 18000
文档：http://127.0.0.1:18000/docs
"""
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from fastapi.middleware.cors import CORSMiddleware

from . import config
from .auth import require_check, require_modify
from .db import FaceStore, ensure_database
from .engine import get_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化数据库并预热模型：避免首个请求承担 ~10s 的模型加载
    ensure_database(config.DATABASE_URL)
    app.state.store = FaceStore(config.DATABASE_URL)
    app.state.engine = get_engine()
    yield
    app.state.store.close()


app = FastAPI(title="Face Recognition Backend", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_image(file: UploadFile) -> bytes:
    """同步读取上传文件（路由为同步 def，跑在线程池中）。"""
    # 优先用声明的 Content-Length 提前拒绝，避免无意义读入
    if file.size is not None and file.size > config.MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"图片超过大小上限 {config.MAX_IMAGE_BYTES} 字节")
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传内容为空")
    if len(data) > config.MAX_IMAGE_BYTES:  # 客户端未声明大小时兜底
        raise HTTPException(status_code=413, detail=f"图片超过大小上限 {config.MAX_IMAGE_BYTES} 字节")
    return data


def _analyze(image: UploadFile):
    """解码并检测，返回 DetectedFace 列表（可为空，由调用方决定如何处理）。"""
    data = _read_image(image)
    try:
        return app.state.engine.analyze(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _analyze_single(image: UploadFile):
    """登记专用：强制恰好一张人脸，防止合影登记时取错人。"""
    faces = _analyze(image)
    if not faces:
        raise HTTPException(status_code=422, detail="未在照片中检测到人脸")
    if len(faces) > 1:
        raise HTTPException(status_code=422, detail=f"登记照片检测到 {len(faces)} 张人脸，无法确定登记对象，请上传单人照片")
    return faces[0]


@app.post("/faces/{user_id}", status_code=201, dependencies=[Depends(require_modify)])
def register_face(user_id: str, image: UploadFile = File(..., description="此人单人正脸照片")):
    """为指定 teamusers 用户登记人脸。重复登记会覆盖旧特征。"""
    user_id = user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id 不能为空")
    if len(user_id) > config.MAX_USER_ID_LEN:
        raise HTTPException(status_code=400, detail=f"user_id 超过长度上限 {config.MAX_USER_ID_LEN}")

    face = _analyze_single(image)
    existed = app.state.store.exists(user_id)
    app.state.store.add(user_id=user_id, embedding=face.embedding)
    return {"user_id": user_id, "overwritten": existed}


@app.get("/faces", dependencies=[Depends(require_check)])
def list_faces():
    ids = app.state.store.list()
    return {"count": len(ids), "user_ids": ids}


@app.get("/faces/{user_id}", dependencies=[Depends(require_check)])
def get_face(user_id: str):
    if not app.state.store.exists(user_id):
        raise HTTPException(status_code=404, detail="该用户未登记人脸")
    return {"user_id": user_id, "registered": True}


@app.delete("/faces/{user_id}", dependencies=[Depends(require_modify)])
def delete_face(user_id: str):
    if not app.state.store.delete(user_id):
        raise HTTPException(status_code=404, detail="该用户未登记人脸")
    return {"deleted": user_id}


@app.post("/recognize", dependencies=[Depends(require_check)])
def recognize(image: UploadFile = File(..., description="待识别照片，可含多张人脸")):
    """识别照片：逐脸返回位置及匹配的 user_id；未匹配时 user_id 为 null。

    图中没有人脸时返回 faces_found=0 与空列表（200，属正常业务结果而非错误）。
    """
    faces = _analyze(image)
    results = []
    for face in faces:
        user_id, similarity = app.state.store.search(face.embedding, config.MATCH_THRESHOLD)
        results.append({
            "bbox": face.bbox,
            "det_score": face.det_score,
            "matched": user_id is not None,
            "similarity": similarity,
            "user_id": user_id,
        })
    return {"faces_found": len(results), "faces": results}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
