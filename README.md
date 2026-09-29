# 人脸识别后端

传入一张照片（可含多张人脸），返回每张人脸的位置及其匹配到的 **teamusers 用户 id**。本服务只存用户 id 与人脸特征，不存姓名等任何其他信息。

## 技术栈

| 组件 | 选型 | 说明 |
|---|---|---|
| 人脸检测 | SCRFD（InsightFace） | ICLR 2022，`det_10g.onnx` |
| 特征提取 | ArcFace R100（InsightFace buffalo_l） | 512 维 L2 归一化特征向量，LFW 99.8%+ |
| 推理运行时 | ONNX Runtime (CPU) | 无需 GPU，无 C++ 编译依赖 |
| Web 框架 | FastAPI + Uvicorn | 自动 OpenAPI 文档（`/docs`） |
| 鉴权 | **teamusers-sdk** | EdDSA 令牌验签（JWKS）+ 权限点判定，本服务不自签令牌 |
| 数据库 | PostgreSQL（psycopg3） | 特征向量存 BYTEA |
| 依赖管理 | **uv**（pyproject.toml） | 现代 Python 项目管理 |

## 鉴权与权限

所有业务接口要求 teamusers 访问令牌（`Authorization: Bearer <token>`），按权限点放行：

| 权限点 | 放行的接口 |
|---|---|
| `face:check:any` | `POST /recognize`、`GET /faces`、`GET /faces/{user_id}` |
| `face:modify:any` | `POST /faces/{user_id}`、`DELETE /faces/{user_id}` |

无令牌 / 令牌无效 → `401`；已认证但无权限 → `403`。`/healthz` 与 `/docs` 无需鉴权。

## 项目结构

```
face_backend/
├── app/
│   ├── __init__.py
│   ├── config.py      # 配置项（环境变量可覆盖）
│   ├── auth.py        # teamusers 验签 + 权限点依赖（face:check/modify:any）
│   ├── engine.py      # InsightFace 推理封装：解码 → 检测 → 特征
│   ├── db.py          # FaceStore：PostgreSQL 存取 + 内存缓存向量检索
│   └── main.py        # FastAPI 路由层（lifespan 预热模型、自动建库建表）
├── pyproject.toml     # uv 项目定义与依赖
├── uv.lock            # 锁定依赖版本
└── .vscode/           # F5 调试、任务、推荐扩展
```

## API 参考

### 登记人脸 `POST /faces/{user_id}`（需 `face:modify:any`）

`user_id` 为 teamusers 规范用户 id（路径参数）。照片必须为单人照（0 张/多张人脸 → 422）。同一 user_id 重复登记会覆盖旧特征。

```bash
curl -X POST http://127.0.0.1:18000/faces/u-1001 \
  -H "Authorization: Bearer <token>" \
  -F "image=@zhangsan.jpg"
```

响应 `201`：`{"user_id": "u-1001", "overwritten": false}`

### 识别人脸 `POST /recognize`（需 `face:check:any`）

一张照片可含多张人脸，逐脸返回结果；图中没有人脸时返回 `faces_found: 0` 与空列表（200）。

```bash
curl -X POST http://127.0.0.1:18000/recognize \
  -H "Authorization: Bearer <token>" \
  -F "image=@photo.jpg"
```

响应 `200`：

```json
{
  "faces_found": 2,
  "faces": [
    {"bbox": [60.5, 40.2, 180.7, 190.9], "det_score": 0.92, "matched": true,  "similarity": 0.71, "user_id": "u-1001"},
    {"bbox": [300.1, 55.0, 410.3, 185.5], "det_score": 0.89, "matched": false, "similarity": 0.21, "user_id": null}
  ]
}
```

| 字段 | 说明 |
|---|---|
| `faces[].bbox` | 人脸框 `[x1, y1, x2, y2]`，相对**原图**坐标 |
| `faces[].det_score` | 检测置信度 0~1 |
| `faces[].similarity` | 与库中最相似人脸的余弦相似度；未匹配时也返回，便于排查 |
| `faces[].user_id` | 匹配到的 teamusers 用户 id；陌生人（相似度 < 阈值）为 `null` |

### 其他接口

| 方法 | 路径 | 权限 | 说明 |
|---|---|---|---|
| `GET` | `/faces` | check | 列出所有已登记的 user_id |
| `GET` | `/faces/{user_id}` | check | 查询是否已登记（未登记 → 404） |
| `DELETE` | `/faces/{user_id}` | modify | 删除登记 |
| `GET` | `/healthz` | 无 | 健康检查 |

## 配置项（`app/config.py`，均可用环境变量覆盖）

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `FACE_TEAMUSERS_URL` | `http://127.0.0.1:8080` | teamusers IAM 地址 |
| `FACE_TEAMUSERS_AUDIENCE` | `teamusers` | 令牌 audience |
| `FACE_TEAMUSERS_SERVICE_TOKEN` | （空） | 服务令牌，拉取权限用，**必须配置** |
| `FACE_DATABASE_URL` | `postgresql://postgres:postgres@127.0.0.1:5432/face_recognition` | PostgreSQL 连接串；目标库不存在时自动创建 |
| `FACE_MODEL_PACK` | `buffalo_l` | 模型包 |
| `FACE_MATCH_THRESHOLD` | `0.45` | 匹配阈值。误识多→调高到 0.5；认不出→调低到 0.4 |
| `FACE_MAX_FACES` | `32` | 单图检测上限 |
| `FACE_MAX_IMAGE_BYTES` | `10485760` | 上传大小上限（10MB） |
| `FACE_MAX_PIXELS` | `40000000` | 解码后像素上限（防像素炸弹） |
| `FACE_CORS_ORIGINS` | `*` | 允许的跨域来源，逗号分隔；生产应收窄 |

## 本地运行

前置：PostgreSQL 可连接；teamusers IAM 可连接。

```bash
uv sync                                # 安装依赖（自动创建 .venv）
uv run uvicorn app.main:app --host 0.0.0.0 --port 18000
```

- 启动时自动完成：建库（`face_recognition`）→ 建表（`faces`）→ 加载模型预热；
- 交互文档：<http://127.0.0.1:18000/docs>；
- 模型包 buffalo_l 在 `~/.insightface/models/`（首次自动下载）。

VS Code：打开本文件夹按 **F5** 调试启动；`Tasks: Run Task` 提供"启动后端 / 运行冒烟测试 / 安装依赖"。

## 测试

冒烟测试（本地保留，不入库）内置打桩 IAM，模拟 admin / viewer / nobody 三种权限：

```bash
set FACE_TEAMUSERS_URL=http://127.0.0.1:18999
set FACE_TEAMUSERS_SERVICE_TOKEN=test-service-token
uv run uvicorn app.main:app --port 18000     # 终端 A
uv run python smoke_test.py                  # 终端 B
```

20 项断言：401/403 鉴权矩阵、viewer 只读、登记、识别本人（sim=1.0）、陌生人拒识、合影多脸识别、多脸登记拒绝 422、无脸 422、非图片 400、像素炸弹 400、CORS、删除后 404 等。

## 设计要点

- **只存 user_id 与特征**：人员信息以 teamusers 为准，本服务零冗余；
- **内存特征缓存**：识别不再每次全表扫描，增删时缓存失效重建，万级人员单次检索 <10ms；
- **登记单人脸约束**：登记照片含多张人脸直接拒绝，杜绝"合影登记取错人"的静默数据污染；
- **启动预热**：模型在 lifespan 阶段加载，`/healthz` 就绪即代表模型可用；
- **线程安全**：推理与数据库访问各由一把锁串行化，FastAPI 线程池下并发安全；
- **防御性输入校验**：文件大小（Content-Length 提前拒绝）+ 解码后像素上限（防像素炸弹）。

## 许可证注意

InsightFace 代码为 MIT；**buffalo 系列模型权重商用需向 insightface.ai 申请授权**（recognition-oss-pack@insightface.ai）。学习与课程项目不受限。
