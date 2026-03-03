import os
import sys
from pathlib import Path

import pymysql
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from video_process.person_search.person_search_engine import PersonSearchEngine


app = FastAPI(title="视频监控轨迹追踪 API", version="1.0.0")
search_engine = PersonSearchEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "219.216.99.30"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "q1w2e3az"),
    "database": os.getenv("MYSQL_DATABASE", "monitor_database"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
    "autocommit": True,
}


DEFAULT_PERSON_CROPS_DIR = PROJECT_ROOT / "person_crops"
VIDEO_PROCESS_PERSON_CROPS_DIR = PROJECT_ROOT / "video_process" / "person_crops"
PERSON_CROPS_DIR = VIDEO_PROCESS_PERSON_CROPS_DIR if VIDEO_PROCESS_PERSON_CROPS_DIR.exists() else DEFAULT_PERSON_CROPS_DIR
PERSON_CROPS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/person_crops", StaticFiles(directory=str(PERSON_CROPS_DIR)), name="person_crops")
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")


class SearchRequest(BaseModel):
    video_id: int
    person_index: int


class TrackingPoint(BaseModel):
    track_id: int
    x: float
    y: float


def get_db_connection():
    return pymysql.connect(**DB_CONFIG)


class TrackingHub:
    def __init__(self):
        self.clients = set()
        self.latest_points = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.clients.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.clients.discard(websocket)

    async def publish_point(self, payload: dict):
        self.latest_points[payload["track_id"]] = payload
        dead_clients = []
        for client in self.clients:
            try:
                await client.send_json(payload)
            except Exception:
                dead_clients.append(client)
        for client in dead_clients:
            self.disconnect(client)


tracking_hub = TrackingHub()


@app.get("/")
def home_page():
    return FileResponse(str(BASE_DIR / "index.html"))


@app.get("/api/videos")
def list_videos(limit: int = 200):
    sql = """
        SELECT id, video_name, created_time
        FROM videos
        ORDER BY id DESC
        LIMIT %s
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall()
        conn.close()
        return {"videos": rows}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"查询 videos 失败: {exc}")


@app.get("/api/videos/{video_id}/persons")
def get_video_persons(video_id: int):
    sql = """
        SELECT person_index, person_image_path
        FROM video_vectors
        WHERE video_id = %s
        ORDER BY person_index ASC
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(sql, (video_id,))
            rows = cursor.fetchall()
        conn.close()

        persons = []
        for row in rows:
            image_path = row["person_image_path"]

            normalized = str(image_path).replace("\\", "/").lstrip("/")
            if normalized.startswith("person_crops/"):
                normalized = normalized[len("person_crops/"):]
            image_url = f"/person_crops/{normalized}"

            persons.append(
                {
                    "person_index": row["person_index"],
                    "person_image_path": row["person_image_path"],
                    "person_image_url": image_url,
                }
            )
        return {"video_id": video_id, "persons": persons}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"查询候选人失败: {exc}")


@app.post("/api/search")
def search_person(body: SearchRequest):
    try:
        results = search_engine.search_target(body.video_id, body.person_index)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"搜索失败: {exc}")

    if not results:
        return {
            "video_id": body.video_id,
            "person_index": body.person_index,
            "matched_count": 0,
            "matches": [],
            "person_trajectory": [],
        }

    first_traj = results[0].get("trajectory", {})
    points = first_traj.get("points", []) if isinstance(first_traj, dict) else []
    return {
        "video_id": body.video_id,
        "person_index": body.person_index,
        "matched_count": len(results),
        "matches": results,
        "person_trajectory": points,
    }


@app.post("/api/search/person")
def search_person_alias(body: SearchRequest):
    return search_person(body)


@app.post("/api/tracking/push")
async def push_tracking_point(body: TrackingPoint):
    payload = {"track_id": body.track_id, "x": body.x, "y": body.y}
    await tracking_hub.publish_point(payload)
    return {"ok": True, "data": payload}


@app.websocket("/ws/tracking")
async def tracking_ws(websocket: WebSocket):
    await tracking_hub.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        tracking_hub.disconnect(websocket)
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
