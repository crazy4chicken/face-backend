"""冒烟测试：注册→识别（单人/合影）→异常输入→删除，走真实 HTTP 接口。

测试素材（tests/assets/）：
- person_a.jpg  单人照 A（注册对象）
- stranger.jpg  单人照 B（陌生人）
- 合影 t1.jpg   验证"多脸注册被拒绝"与"合影识别返回多个结果"

用法：先启动服务（uvicorn app.main:app --port 18000），再运行 python smoke_test.py
"""
import os
import sys

import cv2
import insightface
import numpy as np
import requests

BASE = os.getenv("BASE_URL", "http://127.0.0.1:18000")
ASSETS = os.path.join(os.path.dirname(__file__), "tests", "assets")
PERSON_A = os.path.join(ASSETS, "person_a.jpg")
STRANGER = os.path.join(ASSETS, "stranger.jpg")
GROUP_PHOTO = os.path.join(os.path.dirname(insightface.__file__), "data", "images", "t1.jpg")

failures = []
HEADERS: dict = {}


def check(name: str, ok: bool, detail: str = ""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def post_image(path, file_path, data=None):
    with open(file_path, "rb") as f:
        return requests.post(
            f"{BASE}{path}",
            files={"image": (os.path.basename(file_path), f, "image/jpeg")},
            data=data or {},
            headers=HEADERS,
        )


# 0. 健康检查（无需鉴权）
r = requests.get(f"{BASE}/healthz")
check("健康检查", r.status_code == 200 and r.json()["status"] == "ok", f"-> {r.status_code}")

# 0b. 无 Token 调业务接口应返回 401
r = requests.get(f"{BASE}/persons")
check("无 Token 返回 401", r.status_code == 401, f"-> {r.status_code}")

# 0c. 伪造 Token 应返回 401
r = requests.get(f"{BASE}/persons", headers={"Authorization": "Bearer fake.token.here"})
check("伪造 Token 返回 401", r.status_code == 401, f"-> {r.status_code}")

# 0d. 错误密码登录应返回 401
r = requests.post(f"{BASE}/auth/token", data={"username": "admin", "password": "wrong"})
check("错误密码返回 401", r.status_code == 401, f"-> {r.status_code}")

# 0e. 正确账号密码登录获取 Token
r = requests.post(f"{BASE}/auth/token", data={"username": "admin", "password": "admin123"})
check("登录获取 Token", r.status_code == 200 and "access_token" in r.json(), f"-> {r.status_code}")
HEADERS["Authorization"] = f"Bearer {r.json()['access_token']}"

# 1. 注册单人照 A
r = post_image("/persons", PERSON_A, data={"name": "PersonA", "info": '{"role": "test"}'})
check("注册人员", r.status_code == 201, f"-> {r.status_code} {r.text[:200]}")
person_id = r.json().get("id") if r.status_code == 201 else None

# 2. 识别同一张单人照应匹配
r = post_image("/recognize", PERSON_A)
faces = r.json().get("faces", []) if r.status_code == 200 else []
check(
    "识别已注册人员",
    r.status_code == 200 and len(faces) == 1 and faces[0]["matched"]
    and faces[0]["person"]["name"] == "PersonA" and faces[0]["similarity"] > 0.6,
    f"-> {r.status_code} sim={faces[0].get('similarity') if faces else None}",
)

# 3. 识别陌生人应不匹配
r = post_image("/recognize", STRANGER)
faces = r.json().get("faces", []) if r.status_code == 200 else []
check(
    "陌生人返回 matched=false",
    r.status_code == 200 and len(faces) == 1 and faces[0]["matched"] is False and faces[0]["person"] is None,
    f"-> {r.status_code} sim={faces[0].get('similarity') if faces else None}",
)

# 4. 合影识别：6 张人脸，恰好 1 个匹配、5 个陌生人
r = post_image("/recognize", GROUP_PHOTO)
faces = r.json().get("faces", []) if r.status_code == 200 else []
matched = [f for f in faces if f.get("matched")]
check("合影识别返回多个结果", r.status_code == 200 and len(faces) == 6, f"-> faces_found={len(faces)}")
check(
    "合影中认出已注册者",
    len(matched) == 1 and matched[0]["person"]["name"] == "PersonA",
    f"-> matched={[(m['person']['name'], m['similarity']) for m in matched]}",
)

# 5. 合影注册应被拒绝（单人脸约束）
r = post_image("/persons", GROUP_PHOTO, data={"name": "Multi"})
check("多张人脸注册返回 422", r.status_code == 422, f"-> {r.status_code} {r.text[:120]}")

# 6. 无脸图片注册应返回 422
blank = cv2.imencode(".jpg", np.full((480, 640, 3), 128, np.uint8))[1].tobytes()
r = requests.post(f"{BASE}/persons", files={"image": ("blank.jpg", blank, "image/jpeg")}, data={"name": "Blank"}, headers=HEADERS)
check("无脸图片注册返回 422", r.status_code == 422, f"-> {r.status_code}")

# 7. 无脸图片识别返回空列表（200，正常业务结果）
r = requests.post(f"{BASE}/recognize", files={"image": ("blank.jpg", blank, "image/jpeg")}, headers=HEADERS)
check(
    "无脸识别返回空列表",
    r.status_code == 200 and r.json().get("faces_found") == 0 and r.json().get("faces") == [],
    f"-> {r.status_code} {r.text[:120]}",
)

# 8. 上传非图片应返回 400
r = requests.post(f"{BASE}/recognize", files={"image": ("x.txt", b"not an image", "text/plain")}, headers=HEADERS)
check("非图片返回 400", r.status_code == 400, f"-> {r.status_code}")

# 9. 像素炸弹：小文件解码出 6400 万像素（>4000 万上限）应返回 400
bomb = cv2.imencode(".png", np.full((8000, 8000, 3), 255, np.uint8))[1].tobytes()
r = requests.post(f"{BASE}/recognize", files={"image": ("bomb.png", bomb, "image/png")}, headers=HEADERS)
check("像素炸弹返回 400", r.status_code == 400, f"-> {r.status_code} {r.text[:120]}")

# 10. 空姓名注册应返回 400
r = post_image("/persons", PERSON_A, data={"name": "   "})
check("空姓名返回 400", r.status_code == 400, f"-> {r.status_code}")

# 11. info 不是 JSON 对象应返回 400
r = post_image("/persons", PERSON_A, data={"name": "X", "info": "[1,2,3]"})
check("info 非对象返回 400", r.status_code == 400, f"-> {r.status_code}")

# 12. CORS：带 Origin 的请求应返回允许头
r = requests.get(f"{BASE}/persons", headers={**HEADERS, "Origin": "http://localhost:5173"})
check("CORS 允许跨域", "access-control-allow-origin" in r.headers, f"-> {r.headers.get('access-control-allow-origin')}")

# 13. 人员列表包含已注册者
r = requests.get(f"{BASE}/persons", headers=HEADERS)
check("人员列表", r.status_code == 200 and any(p["name"] == "PersonA" for p in r.json()["persons"]), f"-> {r.text[:200]}")

# 14. 删除人员
if person_id is not None:
    r = requests.delete(f"{BASE}/persons/{person_id}", headers=HEADERS)
    check("删除人员", r.status_code == 200, f"-> {r.status_code}")
    r = requests.get(f"{BASE}/persons/{person_id}", headers=HEADERS)
    check("删除后查询返回 404", r.status_code == 404, f"-> {r.status_code}")

print("\n=== %s ===" % ("全部通过" if not failures else f"{len(failures)} 项失败: {failures}"))
sys.exit(0 if not failures else 1)
