import json
import math
import os
import re
from urllib import error, request
from datetime import datetime

import pymysql
from langchain_core.tools import tool


def _load_mysql_config() -> dict:
	config_path = os.path.join(os.path.dirname(__file__), "config.json")
	with open(config_path, "r", encoding="utf-8") as f:
		config = json.load(f)
	return config.get("mysql", {})


def _cosine_similarity(vec1, vec2) -> float:
	if not isinstance(vec1, list) or not isinstance(vec2, list):
		return 0.0
	if not vec1 or not vec2 or len(vec1) != len(vec2):
		return 0.0

	dot = 0.0
	norm1 = 0.0
	norm2 = 0.0
	for a, b in zip(vec1, vec2):
		try:
			af = float(a)
			bf = float(b)
		except (TypeError, ValueError):
			return 0.0
		dot += af * bf
		norm1 += af * af
		norm2 += bf * bf

	if norm1 <= 0.0 or norm2 <= 0.0:
		return 0.0

	return dot / (math.sqrt(norm1) * math.sqrt(norm2))


def _extract_camera_id(video_name: str) -> str:
	if not isinstance(video_name, str):
		return "unknown"
	match = re.search(r"(camera\d+)", video_name)
	return match.group(1) if match else "unknown"


@tool
def get_video_by_time(start_time: str, end_time: str) -> dict:
	"""
	按时间范围查询 videos 表，返回极简视频清单（id、video_name、created_time）。

	【严格调用边界 - 必须遵守】
	1) 本工具优先级极低，属于粗粒度全局检索工具。
	2) 若用户问题中除了时间，还包含地点线索（如“307厕所”“机房”）或视觉/语义线索（如“蓝衣男子”“摔倒”），
	   绝对禁止优先调用本工具。
	   必须优先调用更精准的工具（地点检索或语义检索），并把时间条件一并传入那些工具。
	3) 仅允许在以下两种场景调用本工具：
	   - 场景 A：用户只问时间段范围，没有任何地点、人物、行为等线索。
	   - 场景 B：作为最后兜底（Fallback），当其他精准工具已尝试且失败、线索中断，需要扩大全盘搜索范围。
	4) 本工具输入必须是精确时间：start_time / end_time 均为 YYYY-MM-DD HH:MM:SS。
	   遇到“前天晚上”“昨天下午两点左右”等模糊时间，Agent 必须先自行推算成精确起止时间再调用本工具。
	   严禁将模糊自然语言原文直接传入。

	Args:
		start_time: 搜索起始时间，格式 YYYY-MM-DD HH:MM:SS。
		end_time: 搜索结束时间，格式 YYYY-MM-DD HH:MM:SS。

	Returns:
		{
			"status": "success/error",
			"message": "提示信息",
			"videos": [
				{"id": 1, "video_name": "cam_01_1400.mp4", "created_time": "2026-02-23 14:00:00"}
			]
		}
	"""
	try:
		start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
		end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
	except ValueError:
		return {
			"status": "error",
			"message": "时间格式错误，请使用 YYYY-MM-DD HH:MM:SS",
			"videos": [],
		}

	if start_dt > end_dt:
		return {
			"status": "error",
			"message": "start_time 不能晚于 end_time",
			"videos": [],
		}

	conn = None
	try:
		mysql_cfg = _load_mysql_config()
		conn = pymysql.connect(
			host=mysql_cfg.get("host"),
			port=int(mysql_cfg.get("port", 3306)),
			user=mysql_cfg.get("user"),
			password=mysql_cfg.get("password"),
			database=mysql_cfg.get("database"),
			charset=mysql_cfg.get("charset", "utf8mb4"),
			autocommit=bool(mysql_cfg.get("autocommit", False)),
			cursorclass=pymysql.cursors.DictCursor,
		)

		with conn.cursor() as cursor:
			sql = """
				SELECT id, video_name, created_time
				FROM videos
				WHERE created_time BETWEEN %s AND %s
				ORDER BY created_time ASC
			"""
			cursor.execute(sql, (start_time, end_time))
			rows = cursor.fetchall()

		videos = [
			{
				"id": row["id"],
				"video_name": row["video_name"],
				"created_time": row["created_time"].strftime("%Y-%m-%d %H:%M:%S")
				if row.get("created_time")
				else None,
			}
			for row in rows
		]

		return {
			"status": "success",
			"message": f"查询成功，共找到 {len(videos)} 条视频记录",
			"videos": videos,
		}
	except Exception as e:
		return {
			"status": "error",
			"message": f"查询失败: {e}",
			"videos": [],
		}
	finally:
		if conn is not None:
			conn.close()


@tool
def get_videos_by_location(
	location_name: str,
	radius: float = 80.0,
	start_time: str = None,
	end_time: str = None,
) -> dict:
	"""
	根据地点名称检索经过该地点附近的行人轨迹，返回匹配的视频与人物索引。

	【极其重要】如果用户的提问中同时包含了时间和地点（例如“昨天下午2点在机房”），
	请务必同时传入 start_time 和 end_time 参数，利用本工具进行高效的时空联合过滤，
	不要单独调用纯时间工具。

	Args:
		location_name: 目标区域名称（例如：机房）。
		radius: 判定半径，默认 80.0。
		start_time: 可选，搜索起始时间，格式 YYYY-MM-DD HH:MM:SS。
		end_time: 可选，搜索结束时间，格式 YYYY-MM-DD HH:MM:SS。

	Returns:
		{
			"status": "success/error",
			"location": "机房",
			"target_coordinates": [2666, 691],
			"matched_count": 2,
			"results": [
				{"video_id": 101, "person_index": 1},
				{"video_id": 105, "person_index": 2}
			]
		}
	"""
	config_path = os.path.join(os.path.dirname(__file__), "config.json")
	with open(config_path, "r", encoding="utf-8") as f:
		config = json.load(f)

	regions = config.get("regions", {})
	if location_name not in regions:
		supported_locations = list(regions.keys())
		return {
			"status": "error",
			"location": location_name,
			"target_coordinates": None,
			"matched_count": 0,
			"results": [],
			"message": (
				f"location_name 不存在: {location_name}。"
				f"当前支持地点: {supported_locations}"
			),
		}

	if radius <= 0:
		return {
			"status": "error",
			"location": location_name,
			"target_coordinates": regions.get(location_name),
			"matched_count": 0,
			"results": [],
			"message": "radius 必须大于 0",
		}

	if (start_time and not end_time) or (end_time and not start_time):
		return {
			"status": "error",
			"location": location_name,
			"target_coordinates": regions.get(location_name),
			"matched_count": 0,
			"results": [],
			"message": "start_time 和 end_time 必须同时传入",
		}

	if start_time and end_time:
		try:
			start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
			end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
		except ValueError:
			return {
				"status": "error",
				"location": location_name,
				"target_coordinates": regions.get(location_name),
				"matched_count": 0,
				"results": [],
				"message": "时间格式错误，请使用 YYYY-MM-DD HH:MM:SS",
			}

		if start_dt > end_dt:
			return {
				"status": "error",
				"location": location_name,
				"target_coordinates": regions.get(location_name),
				"matched_count": 0,
				"results": [],
				"message": "start_time 不能晚于 end_time",
			}

	target_x, target_y = regions[location_name]
	conn = None
	matched = []
	seen = set()

	try:
		mysql_cfg = _load_mysql_config()
		conn = pymysql.connect(
			host=mysql_cfg.get("host"),
			port=int(mysql_cfg.get("port", 3306)),
			user=mysql_cfg.get("user"),
			password=mysql_cfg.get("password"),
			database=mysql_cfg.get("database"),
			charset=mysql_cfg.get("charset", "utf8mb4"),
			autocommit=bool(mysql_cfg.get("autocommit", False)),
			cursorclass=pymysql.cursors.DictCursor,
		)

		with conn.cursor() as cursor:
			if start_time and end_time:
				sql = """
					SELECT vv.video_id, vv.person_index, vv.person_trajectory
					FROM video_vectors vv
					JOIN videos v ON vv.video_id = v.id
					WHERE v.created_time BETWEEN %s AND %s
				"""
				cursor.execute(sql, (start_time, end_time))
			else:
				sql = """
					SELECT video_id, person_index, person_trajectory
					FROM video_vectors
				"""
				cursor.execute(sql)
			rows = cursor.fetchall()

		for row in rows:
			trajectory_text = row.get("person_trajectory")
			if not trajectory_text:
				continue

			try:
				trajectory = json.loads(trajectory_text)
			except json.JSONDecodeError:
				continue

			points = trajectory.get("points", []) if isinstance(trajectory, dict) else []
			for point in points:
				if not isinstance(point, (list, tuple)) or len(point) < 2:
					continue
				x, y = point[0], point[1]
				distance = math.hypot(x - target_x, y - target_y)
				if distance <= radius:
					key = (row.get("video_id"), row.get("person_index"))
					if key not in seen:
						seen.add(key)
						matched.append(
							{
								"video_id": row.get("video_id"),
								"person_index": row.get("person_index"),
							}
						)
					break

		return {
			"status": "success",
			"location": location_name,
			"target_coordinates": [target_x, target_y],
			"matched_count": len(matched),
			"results": matched,
		}
	except Exception as e:
		return {
			"status": "error",
			"location": location_name,
			"target_coordinates": [target_x, target_y],
			"matched_count": 0,
			"results": [],
			"message": f"查询失败: {e}",
		}
	finally:
		if conn is not None:
			conn.close()


@tool
def get_videos_by_semantic(query: str, k: int = 5) -> dict:
	"""
	基于自然语言进行视频语义检索。

	本工具不会在本进程加载向量模型，也不会直连 ES/MySQL，
	仅作为 HTTP Client 调用本地 Flask 语义检索服务：
	- URL: http://127.0.0.1:5000/api/query
	- Method: POST
	- Body: {"query": "用户搜索词", "k": 5}

	Args:
		query: 用户语义搜索文本。
		k: 返回结果数量，默认 5。

	Returns:
		后端返回的 JSON 字典，失败时返回 error 状态与 message。
	"""
	if not isinstance(query, str) or not query.strip():
		return {
			"status": "error",
			"message": "query 不能为空",
			"data": None,
		}

	if not isinstance(k, int) or k <= 0:
		return {
			"status": "error",
			"message": "k 必须是大于 0 的整数",
			"data": None,
		}

	url = "http://127.0.0.1:5000/api/query"
	payload = json.dumps({"query": query.strip(), "k": k}).encode("utf-8")
	headers = {"Content-Type": "application/json"}
	req = request.Request(url=url, data=payload, headers=headers, method="POST")

	try:
		with request.urlopen(req, timeout=30) as resp:
			resp_text = resp.read().decode("utf-8")
			result = json.loads(resp_text)

		if not isinstance(result, dict):
			return {
				"status": "error",
				"message": "语义服务返回格式错误：非 JSON 对象",
				"data": None,
			}

		return result
	except error.HTTPError as e:
		try:
			err_text = e.read().decode("utf-8")
			parsed = json.loads(err_text)
			if isinstance(parsed, dict):
				return parsed
		except Exception:
			pass
		return {
			"status": "error",
			"message": f"语义服务 HTTP 错误: {e.code}",
			"data": None,
		}
	except error.URLError as e:
		return {
			"status": "error",
			"message": f"语义服务不可达: {e.reason}",
			"data": None,
		}
	except json.JSONDecodeError:
		return {
			"status": "error",
			"message": "语义服务返回了无法解析的 JSON",
			"data": None,
		}
	except Exception as e:
		return {
			"status": "error",
			"message": f"调用语义服务失败: {e}",
			"data": None,
		}


@tool
def track_person_globally(video_id: int, person_index: int, start_time: str, end_time: str) -> dict:
	"""
	全局跨镜追踪指定人物：以目标人物特征向量为基准，在给定时间范围内扫描所有摄像头，
	基于余弦相似度筛选可能为同一人的轨迹。

	Args:
		video_id: 目标人物首次出现的视频 ID。
		person_index: 目标人物在该视频中的序号。
		start_time: 追踪起始时间，格式 YYYY-MM-DD HH:MM:SS。
		end_time: 追踪结束时间，格式 YYYY-MM-DD HH:MM:SS。

	Returns:
		{
			"status": "success/error",
			"message": "提示信息",
			"target_info": {"video_id": 101, "person_index": 1},
			"traces": [
				{
					"camera_id": "camera1",
					"video_name": "camera1_20260314_180000",
					"time": "2026-03-14 18:00:00",
					"similarity": 0.85,
					"points": [[2648.4, 436.1], [2651.9, 436.3]]
				}
			]
		}
	"""
	try:
		start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
		end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
	except ValueError:
		return {
			"status": "error",
			"message": "时间格式错误，请使用 YYYY-MM-DD HH:MM:SS",
			"target_info": {"video_id": video_id, "person_index": person_index},
			"traces": [],
		}

	if start_dt > end_dt:
		return {
			"status": "error",
			"message": "start_time 不能晚于 end_time",
			"target_info": {"video_id": video_id, "person_index": person_index},
			"traces": [],
		}

	conn = None
	threshold = 0.70

	try:
		mysql_cfg = _load_mysql_config()
		conn = pymysql.connect(
			host=mysql_cfg.get("host"),
			port=int(mysql_cfg.get("port", 3306)),
			user=mysql_cfg.get("user"),
			password=mysql_cfg.get("password"),
			database=mysql_cfg.get("database"),
			charset=mysql_cfg.get("charset", "utf8mb4"),
			autocommit=bool(mysql_cfg.get("autocommit", False)),
			cursorclass=pymysql.cursors.DictCursor,
		)

		with conn.cursor() as cursor:
			sql_target = """
				SELECT vector_data
				FROM video_vectors
				WHERE video_id = %s AND person_index = %s
				LIMIT 1
			"""
			cursor.execute(sql_target, (video_id, person_index))
			target_row = cursor.fetchone()

			if not target_row or not target_row.get("vector_data"):
				return {
					"status": "error",
					"message": "未找到目标人物向量数据",
					"target_info": {"video_id": video_id, "person_index": person_index},
					"traces": [],
				}

			try:
				target_vec = json.loads(target_row["vector_data"])
			except json.JSONDecodeError:
				return {
					"status": "error",
					"message": "目标人物向量 JSON 解析失败",
					"target_info": {"video_id": video_id, "person_index": person_index},
					"traces": [],
				}

			sql_candidates = """
				SELECT
					v.id AS video_id,
					v.video_name,
					v.created_time,
					vv.person_index,
					vv.vector_data,
					vv.person_trajectory
				FROM videos v
				JOIN video_vectors vv ON v.id = vv.video_id
				WHERE v.created_time BETWEEN %s AND %s
				  AND NOT (vv.video_id = %s AND vv.person_index = %s)
				ORDER BY v.created_time ASC
			"""
			cursor.execute(sql_candidates, (start_time, end_time, video_id, person_index))
			rows = cursor.fetchall()

		traces = []
		for row in rows:
			vector_text = row.get("vector_data")
			if not vector_text:
				continue

			try:
				cand_vec = json.loads(vector_text)
			except json.JSONDecodeError:
				continue

			score = _cosine_similarity(target_vec, cand_vec)
			if score <= threshold:
				continue

			trajectory_text = row.get("person_trajectory")
			points = []
			if trajectory_text:
				try:
					trajectory_obj = json.loads(trajectory_text)
					if isinstance(trajectory_obj, dict):
						parsed_points = trajectory_obj.get("points", [])
						if isinstance(parsed_points, list):
							points = parsed_points
				except json.JSONDecodeError:
					points = []

			created_time = row.get("created_time")
			time_str = (
				created_time.strftime("%Y-%m-%d %H:%M:%S")
				if isinstance(created_time, datetime)
				else str(created_time)
			)

			traces.append(
				{
					"camera_id": _extract_camera_id(row.get("video_name")),
					"video_name": row.get("video_name"),
					"time": time_str,
					"similarity": round(float(score), 4),
					"points": points,
				}
			)

		traces.sort(key=lambda x: x.get("time") or "")

		return {
			"status": "success",
			"message": f"共找到 {len(traces)} 条匹配轨迹",
			"target_info": {"video_id": video_id, "person_index": person_index},
			"traces": traces,
		}
	except Exception as e:
		return {
			"status": "error",
			"message": f"全局追踪失败: {e}",
			"target_info": {"video_id": video_id, "person_index": person_index},
			"traces": [],
		}
	finally:
		if conn is not None:
			conn.close()

