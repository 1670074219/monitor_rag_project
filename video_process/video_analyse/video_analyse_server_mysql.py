import logging
import os
from queue import Queue, Empty
import threading
import base64
import openai
import time
import traceback
import mysql.connector
from mysql.connector import Error
import cv2

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(__file__)

# 数据库配置
DB_CONFIG = {
    'host': '219.216.99.30',
    'port': 3306,
    'database': 'monitor_database',
    'user': 'root',
    'password': 'q1w2e3az',
    'charset': 'utf8mb4',
    'autocommit': False,
}

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.connection = None
        self.db_lock = threading.Lock()
    
    def get_connection(self):
        """获取数据库连接"""
        try:
            if self.connection is None or not self.connection.is_connected():
                self.connection = mysql.connector.connect(**self.config)
            return self.connection
        except Error as e:
            logger.error(f"数据库连接失败: {e}")
            return None
    
    def close_connection(self):
        """关闭数据库连接"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("数据库连接已关闭")

def get_video_duration(video_path: str) -> float:
    """
    获取视频时长（秒）
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        视频时长（秒），如果获取失败返回-1
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return -1
        
        # 获取总帧数和帧率
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        cap.release()
        
        if fps > 0:
            duration = frame_count / fps
            logger.info(f"视频 {os.path.basename(video_path)} 时长: {duration:.2f}秒 ({duration/60:.2f}分钟)")
            return duration
        else:
            logger.error(f"无法获取视频帧率: {video_path}")
            return -1
            
    except Exception as e:
        logger.error(f"获取视频时长失败 {video_path}: {e}")
        return -1

class VideoAnalyServerMySQL(threading.Thread):
    def __init__(self,
                 video_queue: Queue,
                 model_name: str,
                 api_url: str,
                 api_key: str,
                 video_dsp_queue: Queue = None):
        super().__init__(daemon=True)
        self.video_queue = video_queue
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.openai_client = openai.OpenAI(api_key=api_key, base_url=api_url)
        self.video_dsp_queue = video_dsp_queue
        self.db_manager = DatabaseManager(DB_CONFIG)

    def run(self):
        while True:
            try:
                video_record = self.video_queue.get()
                video_id = video_record.get('id') if isinstance(video_record, dict) else None
                video_name = video_record.get('video_name', f"video_{video_id}") if isinstance(video_record, dict) else str(video_record)
                logger.info(f"开始分析: id={video_id}, name={video_name}")
                analyse_result = self.analyze_video(video_record)
                logging.info(f"分析完成: id={video_id}, name={video_name}")
                if analyse_result is not None:
                    self.save_to_database(video_id, analyse_result)
            except Empty:
                time.sleep(1)
                continue
            except Exception as e:
                logger.error(f"处理视频记录时发生错误: {e}")
                logger.debug(traceback.format_exc())
        
    def _handle_problematic_video(self, video_id: int):
        """处理有问题的视频：删除文件和数据库记录"""
        with self.db_manager.db_lock:
            try:
                connection = self.db_manager.get_connection()
                if not connection:
                    logger.error("无法连接数据库")
                    return

                cursor = connection.cursor()
                
                # 1. 获取视频信息
                select_sql = "SELECT video_path FROM videos WHERE id = %s"
                cursor.execute(select_sql, (video_id,))
                result = cursor.fetchone()
                
                if not result:
                    logger.warning(f"无法在数据库中找到视频记录: id={video_id}")
                    cursor.close()
                    return

                video_path = result[0]
                
                # 2. 删除视频文件
                if video_path and os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                        logger.info(f"已删除有问题的视频文件: {video_path}")
                    except OSError as e:
                        logger.error(f"删除视频文件 {video_path} 失败: {e}")
                
                # 3. 删除数据库记录
                delete_sql = "DELETE FROM videos WHERE id = %s"
                cursor.execute(delete_sql, (video_id,))
                connection.commit()
                cursor.close()
                
                logger.info(f"已从数据库中删除视频记录: id={video_id}")

            except Error as e:
                logger.error(f"处理有问题的视频 id={video_id} 时数据库操作失败: {e}")
                if connection:
                    connection.rollback()
            except Exception as e:
                logger.error(f"处理有问题的视频 id={video_id} 时发生未知错误: {e}")
                logger.debug(traceback.format_exc())

    def get_video_info(self, video_id: int):
        """从数据库获取单个视频的信息"""
        with self.db_manager.db_lock:
            try:
                connection = self.db_manager.get_connection()
                if not connection:
                    logger.error("无法连接数据库")
                    return None

                cursor = connection.cursor(dictionary=True)
                select_sql = "SELECT id, video_name, video_path, description FROM videos WHERE id = %s"
                cursor.execute(select_sql, (video_id,))
                result = cursor.fetchone()
                cursor.close()
                
                return result
                
            except Error as e:
                logger.error(f"获取视频信息失败: {e}")
                return None

    def mark_video_as_skipped(self, video_id: int, reason: str):
        """将视频标记为跳过分析"""
        with self.db_manager.db_lock:
            try:
                connection = self.db_manager.get_connection()
                if not connection:
                    logger.error("无法连接数据库")
                    return False

                cursor = connection.cursor()
                
                # 更新分析结果为跳过原因
                skip_message = f"跳过分析: {reason}"
                update_sql = """
                UPDATE videos 
                SET description = %s 
                WHERE id = %s
                """
                cursor.execute(update_sql, (skip_message, video_id))
                connection.commit()
                cursor.close()
                
                logger.info(f"已标记视频 id={video_id} 为跳过: {reason}")
                return True
                
            except Error as e:
                logger.error(f"标记跳过视频失败: {e}")
                if connection:
                    connection.rollback()
                return False

    def smart_extract_keyframes(self,
                                video_path: str,
                                max_frames: int = 12,
                                scan_stride: int = 3,
                                min_interval_sec: float = 1.5,
                                jpeg_quality: int = 62):
        """
        智能关键帧提取（严格流式，避免整视频入内存）。

        算法流程：
        1) 第一遍快速扫描：逐帧读取，但每隔 scan_stride 帧计算一次运动得分；
           得分=前后灰度模糊帧差二值化后的白点数量。
        2) 按运动得分降序做时间NMS：限制最小时间间隔 min_interval_sec，避免帧过于集中。
        3) 按时间升序重排选中帧索引。
        4) 第二遍精确读取：cap.set 到对应帧，读取原始图，缩放 1/2，JPEG 压缩后转 Base64。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频进行抽帧: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        motion_candidates = []
        prev_gray = None
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % scan_stride != 0:
                frame_idx += 1
                continue

            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                _, motion_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
                motion_score = int(cv2.countNonZero(motion_mask))
                motion_candidates.append((motion_score, frame_idx, frame_idx / fps))

            prev_gray = gray
            frame_idx += 1

        cap.release()

        if not motion_candidates:
            logger.warning(f"未提取到运动候选帧: {video_path}")
            return []

        # 得分降序，优先选动作剧烈帧
        motion_candidates.sort(key=lambda x: x[0], reverse=True)

        selected = []
        selected_times = []
        for score, idx, ts in motion_candidates:
            if score <= 0:
                continue
            if any(abs(ts - st) < min_interval_sec for st in selected_times):
                continue
            selected.append(idx)
            selected_times.append(ts)
            if len(selected) >= max_frames:
                break

        # 若全部分数都很低，至少兜底取最高分帧
        if not selected:
            selected = [motion_candidates[0][1]]

        # 时间轴重组
        selected = sorted(set(selected))

        # 第二遍精准读取原始帧并编码
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"二次读取视频失败: {video_path}")
            return []

        b64_images = []
        for idx in selected:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # 降低token与内存：长宽缩小到1/2
            frame_small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_AREA)

            ok_enc, buf = cv2.imencode('.jpg', frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            if not ok_enc:
                continue

            b64_images.append(base64.b64encode(buf.tobytes()).decode('utf-8'))

        cap.release()
        logger.info(f"智能抽帧完成: {os.path.basename(video_path)} -> {len(b64_images)} 帧")
        return b64_images

    def analyze_video(self, video_record: dict):
        """
        分析单个视频记录。

        入参：
            video_record: {'id': int, 'video_path': str, 'video_name': str(可选)}
        """
        if not isinstance(video_record, dict):
            logger.error(f"非法视频记录类型: {type(video_record)}")
            return None

        video_id = video_record.get('id')
        video_path = video_record.get('video_path')
        video_name = video_record.get('video_name', f"video_{video_id}")

        if video_id is None or not video_path:
            logger.error(f"视频记录字段缺失: {video_record}")
            return None

        if not os.path.exists(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            self.mark_video_as_skipped(video_id, "视频文件不存在")
            return None

        duration = get_video_duration(video_path)
        if duration == -1:
            self.mark_video_as_skipped(video_id, "无法获取视频时长")
            return None

        max_duration_sec = 600  # 10分钟
        if duration > max_duration_sec:
            self.mark_video_as_skipped(video_id, f"视频时长超过{max_duration_sec/60:.0f}分钟")
            return None

        # 核心改造：运动强度智能抽帧，避免等距采样漏关键动作
        keyframes_b64 = self.smart_extract_keyframes(
            video_path=video_path,
            max_frames=12,
            scan_stride=3,
            min_interval_sec=1.5,
            jpeg_quality=62
        )

        if not keyframes_b64:
            self.mark_video_as_skipped(video_id, "抽帧失败或无有效关键帧")
            return None

        prompt = f"""你是一个专业的监控视频安全分析专家。请分析以下截取的监控视频帧。
                    请以人为主体，环境为辅，详细且客观地描述视频中发生的事情。如果视频中没有出现任何人，请直接输出：“未见人员活动”。

                    为了方便录入安防数据库，请你**严格遵守**以下层级结构和键值对格式进行输出，不要改变小标题的名称，也不要遗漏任何字段。如果由于画质原因某个特征无法看清，请明确填写“未知”或“模糊不清”。

                    【人数统计】
                    - 总人数：[仅输出阿拉伯数字，如：2]

                    【人员外观特征详情】（请按视频中人物的主次顺序逐一填写）
                    --- 人物 1 ---
                    * 性别/年龄段：[如：男性，青年]
                    * 发型特征：[格式：长度 + 颜色 + 款式/是否有帽子。如：黑色短发 / 戴黑色棒球帽看不到头发]
                    * 上半身着装：[格式：颜色 + 款式。如：红色短袖T恤 / 黑色连帽卫衣]
                    * 下半身着装：[格式：颜色 + 款式。如：蓝色长款牛仔裤 / 黑色短裤]
                    * 鞋履/配饰：[如：白色运动鞋，背着黑色双肩包，戴口罩]

                    --- 人物 2 --- 
                    [如果有第二个人，复制上述5个字段格式进行填写；如果没有，请忽略此部分]
                    ...（以此类推）

                    【行为与时间线】
                    - [描述人物1的具体动作、移动轨迹以及他与其他人的互动]
                    - [描述人物2的具体动作...]

                    【环境与交互】
                    - 场景描述：[简单描述监控所处的环境类型，如街道、走廊、仓库]
                    - 物品交互：[详细描述人员是否触摸、携带、破坏或使用了环境中的什么物体。若无，请写“无明显物理交互”]

                    【高密检索关键词】(非常重要，用于向量搜索)
                    - [请提取 5-10 个高度概括的词组，涵盖颜色、动作、物品、场景，用逗号分隔。例如：红衣男子, 黑色短发, 奔跑, 黑色背包, 翻越栅栏, 夜晚]

                    【结构化总结】
                    摄像头 {video_id} 在 {video_name} 的监控摘要：[用 50 字左右的高度凝练语言总结核心事件，例如：“一名黑色短发、穿红色T恤的男子在仓库门口徘徊后，试图拉开卷帘门。”]
                    """

        # OpenAI标准多图格式：image_url 列表
        user_content = [{"type": "text", "text": prompt}]
        for b64_str in keyframes_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_str}"
                }
            })

        messages = [
            {
                "role": "system",
                "content": "你是一个专业的监控视频分析专家，你的分析应该以人为主体，环境为辅助。"
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                max_tokens=2048,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"调用视觉模型失败(id={video_id}): {e}")
            self.mark_video_as_skipped(video_id, f"模型调用失败: {e}")
            return None
    
    def save_to_database(self, v_k: str, analyse_result: str):
        """将分析结果保存到数据库"""
        with self.db_manager.db_lock:
            try:
                connection = self.db_manager.get_connection()
                if not connection:
                    logger.error("无法连接数据库")
                    return False

                cursor = connection.cursor()
                
                # 更新分析结果
                update_sql = """
                UPDATE videos 
                SET description = %s 
                WHERE id = %s
                """
                cursor.execute(update_sql, (analyse_result, v_k))
                connection.commit()
                cursor.close()
                
                logger.info(f"已更新数据库中 id={v_k} 的分析结果")

                # 如果有embedding队列，将视频加入队列
                if self.video_dsp_queue is not None:
                    self.video_dsp_queue.put(v_k)
                    logger.info(f"已放入embedding队列:id={v_k}")
                
                return True
                
            except Error as e:
                logger.error(f"保存分析结果到数据库失败: {e}")
                if connection:
                    connection.rollback()
                return False

    def __del__(self):
        """析构函数，关闭数据库连接"""
        if hasattr(self, 'db_manager'):
            self.db_manager.close_connection()

def get_unprocessed_videos_from_db(max_count: int = 5):
    """从数据库获取未处理的视频列表"""
    db_manager = DatabaseManager(DB_CONFIG)
    unprocessed_videos = []
    
    try:
        connection = db_manager.get_connection()
        if not connection:
            logger.error("无法连接数据库")
            return unprocessed_videos

        cursor = connection.cursor()
        
        # 查询 videos 表中未分析的视频（description 为空或NULL）
        select_sql = """
        SELECT id, video_name, video_path 
        FROM videos 
        WHERE video_path IS NOT NULL 
        AND (description IS NULL OR description = '') 
        ORDER BY id ASC
        LIMIT %s
        """
        
        cursor.execute(select_sql, (max_count,))
        results = cursor.fetchall()
        
        unprocessed_videos = [
            {
                'id': row[0],
                'video_name': row[1],
                'video_path': row[2],
            }
            for row in results
        ]
        cursor.close()
        
        logger.info(f"从数据库找到 {len(unprocessed_videos)} 个未处理的视频")
        
    except Error as e:
        logger.error(f"从数据库获取未处理视频失败: {e}")
    finally:
        db_manager.close_connection()
    
    return unprocessed_videos

def get_database_stats():
    """获取数据库统计信息"""
    db_manager = DatabaseManager(DB_CONFIG)
    
    try:
        connection = db_manager.get_connection()
        if not connection:
            logger.error("无法连接数据库")
            return

        cursor = connection.cursor()
        
        # 统计总记录数
        cursor.execute("SELECT COUNT(*) FROM videos")
        total_count = cursor.fetchone()[0]
        
        # 统计已分析的记录数（不包括跳过的）
        cursor.execute("SELECT COUNT(*) FROM videos WHERE description IS NOT NULL AND description != '' AND description NOT LIKE '跳过分析:%'")
        analyzed_count = cursor.fetchone()[0]
        
        # 统计跳过的记录数
        cursor.execute("SELECT COUNT(*) FROM videos WHERE description LIKE '跳过分析:%'")
        skipped_count = cursor.fetchone()[0]
        
        # 统计真正未分析的记录数
        cursor.execute("SELECT COUNT(*) FROM videos WHERE description IS NULL OR description = ''")
        unanalyzed_count = cursor.fetchone()[0]
        
        # 统计不同跳过原因
        cursor.execute("""
            SELECT description, COUNT(*) as count 
            FROM videos 
            WHERE description LIKE '跳过分析:%' 
            GROUP BY description 
            ORDER BY count DESC
        """)
        skip_reasons = cursor.fetchall()
        
        cursor.close()
        
        logger.info(f"📊 数据库统计信息:")
        logger.info(f"  总记录数: {total_count}")
        logger.info(f"  ✅ 已分析: {analyzed_count}")
        logger.info(f"  ⏸️ 跳过分析: {skipped_count}")
        logger.info(f"  ⏳ 待分析: {unanalyzed_count}")
        
        if total_count > 0:
            progress = (analyzed_count / total_count) * 100
            logger.info(f"  📈 分析进度: {progress:.2f}%")
        
        if skip_reasons:
            logger.info(f"  🚫 跳过原因统计:")
            for reason, count in skip_reasons:
                logger.info(f"    - {reason}: {count} 个视频")
        
    except Error as e:
        logger.error(f"获取数据库统计信息失败: {e}")
    finally:
        db_manager.close_connection()

if __name__ == "__main__":
    # 仅在主入口局部导入 Full，避免改动全局 import 结构
    from queue import Full

    # 启动时打印数据库统计，便于观察服务初始状态
    get_database_stats()

    # 1) 使用有界队列，避免一次性堆积过多任务导致内存风险
    video_queue = Queue(maxsize=100)

    # 2) 启动消费者线程（后台常驻）
    video_analyse_server = VideoAnalyServerMySQL(
        video_queue,
        "qwen2.5",
        "http://localhost:8000/v1",
        "token-abc123"
    )
    video_analyse_server.start()

    # 3) 主线程作为生产者，按水位线持续轮询数据库补货
    LOW_WATERMARK = 20   # 队列低于该水位才向数据库拉新任务
    FETCH_BATCH = 50     # 每次最多拉取任务数

    logger.info("生产者-消费者常驻服务已启动，按水位线持续拉取未处理视频。")

    try:
        while True:
            current_qsize = video_queue.qsize()

            # 只有队列低水位时才“进货”，避免队列堆积
            if current_qsize < LOW_WATERMARK:
                new_records = get_unprocessed_videos_from_db(max_count=FETCH_BATCH)

                # 数据库暂无新任务：适当休眠，避免空轮询占CPU
                if not new_records:
                    logger.info("数据库暂无新视频任务，10秒后重试。")
                    time.sleep(10)
                    continue

                # 将新任务放入有界队列，防止阻塞死锁
                for record in new_records:
                    try:
                        video_queue.put(record, block=True, timeout=1)
                    except Full:
                        logger.warning("任务队列已满，停止本轮入队，5秒后重试。")
                        break

                logger.info(f"本轮补货完成，当前队列任务数: {video_queue.qsize()}")

                # 有新任务入队后短暂休眠，给消费者处理时间
                time.sleep(1)
            else:
                # 队列仍较高，暂停拉取，避免CPU空转与数据库压力
                logger.info(f"队列水位较高({current_qsize})，5秒后再检查。")
                time.sleep(5)

    except KeyboardInterrupt:
        # 4) 支持 Ctrl+C 优雅退出
        logger.info("接收到 Ctrl+C，主线程准备安全退出。")