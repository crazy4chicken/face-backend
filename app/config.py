"""全局配置：可用环境变量覆盖。"""
import os

# InsightFace 模型包（首次运行自动下载到 ~/.insightface）
MODEL_PACK = os.getenv("FACE_MODEL_PACK", "buffalo_l")

# 检测输入尺寸，越大越容易检出小脸，速度越慢
DET_SIZE = (640, 640)

# 余弦相似度阈值：ArcFace(buffalo_l) 常用 0.4~0.5，越大越严格
MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.45"))

# 单张图最多返回的人脸数
MAX_FACES = int(os.getenv("FACE_MAX_FACES", "32"))

# PostgreSQL 连接串
DATABASE_URL = os.getenv(
    "FACE_DATABASE_URL",
    "postgresql://postgres:postgres@127.0.0.1:5432/face_recognition",
)

# 上传图片大小上限（字节）
MAX_IMAGE_BYTES = int(os.getenv("FACE_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
# 解码后像素总数上限（防像素炸弹 DoS：小文件可解码出几亿像素）
MAX_PIXELS = int(os.getenv("FACE_MAX_PIXELS", str(40_000_000)))

# 允许的跨域来源，逗号分隔；默认 "*" 方便联调，生产环境应收窄
CORS_ORIGINS = [o.strip() for o in os.getenv("FACE_CORS_ORIGINS", "*").split(",") if o.strip()]

# 姓名最大长度
MAX_NAME_LEN = 128

# info JSON 最大长度
MAX_INFO_LEN = 8192

# JWT 鉴权
JWT_SECRET = os.getenv("FACE_JWT_SECRET", "dev-secret-change-me-in-production")
JWT_ALGORITHM = os.getenv("FACE_JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("FACE_JWT_EXPIRE_MINUTES", "120"))

# 登录账号（生产环境必须用环境变量覆盖）
ADMIN_USERNAME = os.getenv("FACE_ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("FACE_ADMIN_PASSWORD", "admin123")
