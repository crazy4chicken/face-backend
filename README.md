# 人脸识别后端

传入一张照片（可含多张人脸），返回每张人脸的位置及其在人员库中匹配到的身份信息。支持人员注册、查询、删除。

## 技术栈

| 组件 | 选型 | 说明 |
|---|---|---|
| 人脸检测 | SCRFD（InsightFace） | ICLR 2022，`det_10g.onnx` |
| 特征提取 | ArcFace R100（InsightFace buffalo_l） | 512 维 L2 归一化特征向量，LFW 99.8%+ |
| 推理运行时 | ONNX Runtime (CPU) | 无需 GPU，无 C++ 编译依赖 |
| Web 框架 | FastAPI + Uvicorn | 自动 OpenAPI 文档（`/docs`） |
| 数据库 | **PostgreSQL 17**（psycopg3） | `info` 用 JSONB 存扩展字段；特征向量存 BYTEA |

## 项目结构

```
face_backend/
├── app/
│   ├── __init__.py
│   ├── config.py      # 配置项（环境变量可覆盖）
│   ├── auth.py        # JWT 签发与校验（HS256）
│   ├── engine.py      # InsightFace 推理封装：解码 → 检测 → 特征
│   ├── db.py          # PersonStore：PostgreSQL 增删查 + 内存缓存向量检索
│   └── main.py        # FastAPI 路由层（lifespan 预热模型、自动建库建表）
├── smoke_test.py      # 冒烟测试（16 项断言，含合影多脸识别）
├── tests/assets/      # 单人照测试素材
├── requirements.txt
└── .vscode/           # F5 调试、任务、推荐扩展
```

## 工作流程

```
上传照片
   │
   ▼
大小/像素校验（超限 → 413/400）
   │
   ▼
cv2 解码（非法图片 → 400）
   │
   ▼
SCRFD 检测所有人脸
   │
   ├── 注册接口：恰好 1 张脸才可注册（0 张/多张 → 422，防止合影注册错人）
   └── 识别接口：逐脸处理；0 张脸 → 200 返回空列表
   │
   ▼
ArcFace 提取 512 维特征（L2 归一化）
   │
   ▼
与内存缓存的库中特征矩阵做点积（= 余弦相似度），取最大值
   │
   ├── 相似度 ≥ 0.45 → 返回人员信息
   └── 相似度 < 0.45 → person=null，仍返回最接近的相似度
```


## 鉴权（JWT）

除 `/healthz`、`/auth/token` 与文档页外，所有接口必须携带 JWT Bearer Token，否则返回 `401`。

```bash
# 1. 登录获取 Token（默认账号 admin/admin123，见配置项，生产必须改）
curl -X POST http://127.0.0.1:18000/auth/token -F "username=admin" -F "password=admin123"
# → {"access_token": "eyJ...", "token_type": "bearer", "expires_in": 7200}

# 2. 携带 Token 调用业务接口
curl -X POST http://127.0.0.1:18000/recognize \
  -H "Authorization: Bearer eyJ..." \
  -F "image=@photo.jpg"
```

签名算法 HS256；Token 过期、伪造、缺失均返回 401 并带 `WWW-Authenticate: Bearer` 头。`/docs` 页面右上角 "Authorize" 按钮填入 Token 后即可在线调试受保护接口。
## API 参考

### 注册人员 `POST /persons`

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | string | 必填，姓名（≤128 字符） |
| `info` | string(JSON 对象) | 选填，附加信息，如 `{"student_id": "001"}` |
| `image` | file | 必填，此人**单人**照片 |

```bash
curl -X POST http://127.0.0.1:18000/persons \
  -F "name=张三" \
  -F 'info={"student_id": "001", "department": "计算机学院"}' \
  -F "image=@zhangsan.jpg"
```

响应 `201`：

```json
{"id": 1, "name": "张三", "info": {"student_id": "001"}, "message": "注册成功"}
```

> 照片含 0 张或多张人脸均返回 `422`（单人脸强约束，防止合影注册错人）。

### 识别人脸 `POST /recognize`

一张照片可含多张人脸，逐脸返回结果；图中没有人脸时返回 `faces_found: 0` 与空列表（200）。

```bash
curl -X POST http://127.0.0.1:18000/recognize -F "image=@photo.jpg"
```

响应 `200`：

```json
{
  "faces_found": 2,
  "faces": [
    {
      "bbox": [60.5, 40.2, 180.7, 190.9],
      "det_score": 0.92,
      "matched": true,
      "similarity": 0.71,
      "person": {
        "id": 1,
        "name": "张三",
        "info": {"student_id": "001"},
        "created_at": "2026-09-24T19:19:04.919484+08:00"
      }
    },
    {
      "bbox": [300.1, 55.0, 410.3, 185.5],
      "det_score": 0.89,
      "matched": false,
      "similarity": 0.21,
      "person": null
    }
  ]
}
```

| 字段 | 说明 |
|---|---|
| `faces[].bbox` | 人脸框 `[x1, y1, x2, y2]`，相对**原图**坐标 |
| `faces[].det_score` | 检测置信度 0~1 |
| `faces[].similarity` | 与库中最相似人员的余弦相似度；未匹配时也返回，便于排查 |
| `faces[].person` | 匹配到的人员信息；陌生人（相似度 < 阈值）为 `null` |

### 其他接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/persons` | 列出所有人员 |
| `GET` | `/persons/{id}` | 查询单人（不存在 → 404） |
| `DELETE` | `/persons/{id}` | 删除人员 |
| `GET` | `/healthz` | 健康检查 |

## 配置项（`app/config.py`，均可用环境变量覆盖）

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `FACE_DATABASE_URL` | `postgresql://postgres:postgres@127.0.0.1:5432/face_recognition` | PostgreSQL 连接串；目标库不存在时自动创建 |
| `FACE_MODEL_PACK` | `buffalo_l` | 模型包；`buffalo_s` 更快但略降精度 |
| `FACE_MATCH_THRESHOLD` | `0.45` | 匹配阈值。误识多→调高到 0.5；认不出→调低到 0.4 |
| `FACE_MAX_FACES` | `32` | 单图检测上限（超过按多脸拒绝） |
| `FACE_MAX_IMAGE_BYTES` | `10485760` | 上传大小上限（10MB） |
| `FACE_MAX_PIXELS` | `40000000` | 解码后像素上限（防像素炸弹） |
| `FACE_CORS_ORIGINS` | `*` | 允许的跨域来源，逗号分隔；生产应收窄 |
| `FACE_JWT_SECRET` | `dev-secret-change-me-in-production` | JWT 签名密钥，**生产必须改** |
| `FACE_JWT_EXPIRE_MINUTES` | `120` | Token 有效期（分钟） |
| `FACE_ADMIN_USERNAME` | `admin` | 登录账号，**生产必须改** |
| `FACE_ADMIN_PASSWORD` | `admin123` | 登录密码，**生产必须改** |

## 本地运行

前置：本机或远程有可用的 PostgreSQL（默认连 `127.0.0.1:5432`，账号 `postgres/postgres`）。

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 18000
```

- 启动时自动完成：建库（`face_recognition`）→ 建表（`persons`）→ 加载模型预热；
- 交互文档：<http://127.0.0.1:18000/docs>；
- 模型包 buffalo_l 在 `~/.insightface/models/`（首次自动下载；GitHub 慢时可用 ghproxy 镜像手动放置解压）。

VS Code：打开本文件夹按 **F5** 调试启动；`Tasks: Run Task` 提供"启动后端 / 运行冒烟测试 / 安装依赖"。

## 测试

```bash
python smoke_test.py
```

16 项断言：健康检查、注册、识别本人（sim=1.0）、陌生人拒识（sim=-0.09）、**合影识别返回 6 张脸且认出已注册者**、多脸注册拒绝 422、无脸注册 422、无脸识别返回空列表、非图片 400、像素炸弹 400、空姓名 400、info 非对象 400、CORS、列表、删除、删除后 404。

## 设计要点

- **单人脸契约**：注册与识别都要求照片恰好一张人脸，从源头杜绝"合影注册错人"的静默错误；
- **内存特征缓存**：库中全部特征在内存中维护为矩阵，增删时失效重建；识别不再每次全表扫描，万级人员单次检索 <10ms；
- **启动预热**：模型在 lifespan 阶段加载，服务 ready 即可全速响应，首请求不承担模型加载耗时；
- **线程安全**：推理与数据库访问各由一把锁串行化，FastAPI 线程池下并发安全；
- **只存特征不存照片**：每人 2KB 向量，规避原始照片泄露风险。

## 许可证注意

InsightFace 代码为 MIT；**buffalo 系列模型权重商用需向 insightface.ai 申请授权**（recognition-oss-pack@insightface.ai）。学习与课程项目不受限。
