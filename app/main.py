"""人脸识别后端 API。

约定：注册照片必须为单人照；识别照片可含多张人脸（通常不多）。

路由（除 /healthz 与 /auth/token 外均需 JWT Bearer Token）：
    POST   /auth/token       登录签发 JWT（账号密码由环境变量配置）
    POST   /persons          注册人员（姓名 + 单人照片；照片含多张人脸时拒绝）
    GET    /persons          列出所有已注册人员
    GET    /persons/{id}     查询单个人员
    DELETE /persons/{id}     删除人员
    POST   /recognize        识别照片中的所有人脸，逐脸返回匹配的人员信息
    GET    /healthz          健康检查（无需鉴权，供探活）

启动：uvicorn app.main:app --host 0.0.0.0 --port 18000
文档：http://127.0.0.1:18000/docs
"""
import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .auth import create_token, require_token, verify_credentials
from .db import PersonStore, ensure_database
from .engine import get_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化数据库并预热模型：避免首个请求承担 ~10s 的模型加载
    ensure_database(config.DATABASE_URL)
    app.state.store = PersonStore(config.DATABASE_URL)
    app.state.engine = get_engine()
    yield
    app.state.store.close()


app = FastAPI(title="Face Recognition Backend", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/auth/token")
def login(
    username: str = Form(...),
    password: str = Form(...),
):
    """账号密码登录，签发 JWT。账号密码由环境变量 FACE_ADMIN_USERNAME/PASSWORD 配置。"""
    if not verify_credentials(username, password):
        raise HTTPException(status_code=401, detail="账号或密码错误")
    return {
        "access_token": create_token(username),
        "token_type": "bearer",
        "expires_in": config.JWT_EXPIRE_MINUTES * 60,
    }


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
    """注册专用：强制恰好一张人脸，防止合影注册时取错人。"""
    faces = _analyze(image)
    if not faces:
        raise HTTPException(status_code=422, detail="未在照片中检测到人脸")
    if len(faces) > 1:
        raise HTTPException(status_code=422, detail=f"注册照片检测到 {len(faces)} 张人脸，无法确定注册对象，请上传单人照片")
    return faces[0]


@app.post("/persons", status_code=201, dependencies=[Depends(require_token)])
def register_person(
    name: str = Form(..., description="姓名"),
    info: str = Form("{}", description='附加信息 JSON 对象，如 {"student_id": "001"}'),
    image: UploadFile = File(..., description="此人单人正脸照片"),
):
    """注册人员。照片必须恰好包含一张人脸，否则返回 422。"""
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="姓名不能为空")
    if len(name) > config.MAX_NAME_LEN:
        raise HTTPException(status_code=400, detail=f"姓名超过长度上限 {config.MAX_NAME_LEN}")
    if len(info) > config.MAX_INFO_LEN:
        raise HTTPException(status_code=400, detail=f"info 超过长度上限 {config.MAX_INFO_LEN}")
    try:
        extra = json.loads(info)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="info 不是合法的 JSON")
    if not isinstance(extra, dict):
        raise HTTPException(status_code=400, detail="info 必须是 JSON 对象")

    face = _analyze_single(image)
    person_id = app.state.store.add(name=name, info=extra, embedding=face.embedding)
    return {"id": person_id, "name": name, "info": extra, "message": "注册成功"}


@app.get("/persons", dependencies=[Depends(require_token)])
def list_persons():
    persons = app.state.store.list()
    return {"count": len(persons), "persons": persons}


@app.get("/persons/{person_id}", dependencies=[Depends(require_token)])
def get_person(person_id: int):
    person = app.state.store.get(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="人员不存在")
    return person


@app.delete("/persons/{person_id}", dependencies=[Depends(require_token)])
def delete_person(person_id: int):
    if not app.state.store.delete(person_id):
        raise HTTPException(status_code=404, detail="人员不存在")
    return {"deleted": person_id}


@app.post("/recognize", dependencies=[Depends(require_token)])
def recognize(image: UploadFile = File(..., description="待识别照片，可含多张人脸")):
    """识别照片：返回每张人脸的位置及匹配结果；未匹配时 person 为 null。

    图中没有人脸时返回 faces_found=0 与空列表（200，属正常业务结果而非错误）。
    """
    faces = _analyze(image)
    results = []
    for face in faces:
        person, similarity = app.state.store.search(face.embedding, config.MATCH_THRESHOLD)
        results.append({
            "bbox": face.bbox,
            "det_score": face.det_score,
            "matched": person is not None,
            "similarity": similarity,
            "person": person,
        })
    return {"faces_found": len(results), "faces": results}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
