# 讲json文件数据导入到mysql数据库中

import json
import pymysql
import re
from datetime import datetime

# ================= 配置区域 (请修改这里) =================
DB_HOST = '219.216.99.30'
DB_PORT = 3306
DB_USER = 'root'
DB_PASSWORD = 'q1w2e3az'
DB_NAME = 'monitor_database'
TABLE_NAME = 'videos'
JSON_FILE = '../video_description_back.json'  # 你的JSON文件路径
# =======================================================

def get_db_connection():
    """连接数据库"""
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

def parse_time_from_name(video_name):
    """
    从文件名提取时间
    输入: camera1_20250623_094842
    输出: 2025-06-23 09:48:42
    """
    # 使用正则匹配 YYYYMMDD_HHMMSS 格式
    match = re.search(r'(\d{8})_(\d{6})', video_name)
    if match:
        date_part = match.group(1) # 20250623
        time_part = match.group(2) # 094842
        try:
            dt = datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S") # 匹配失败则返回当前时间

def extract_person_count(text):
    """
    从描述文本中提取人数
    输入: "人数：2\n\n每个人的外貌特征..."
    输出: 2
    """
    if not text:
        return 0
    # 匹配 "人数：数字" 或 "人数:数字"
    match = re.search(r'人数[:：]\s*(\d+)', text)
    if match:
        return int(match.group(1))
    return 0

def main():
    # 1. 读取 JSON 文件
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取 JSON 文件，共 {len(data)} 条数据。")
    except FileNotFoundError:
        print(f"错误：找不到文件 {JSON_FILE}")
        return

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 准备 SQL 语句
            sql = f"""
                INSERT INTO {TABLE_NAME} 
                (video_name, video_path, person_count, description, created_time) 
                VALUES (%s, %s, %s, %s, %s)
            """
            
            success_count = 0
            
            # 2. 遍历 JSON 数据
            for video_name, info in data.items():
                video_path = info.get('video_path', '')
                description = info.get('analyse_result', '')
                
                # 处理时间 (从文件名 camera1_20250623_094842 提取)
                created_time = parse_time_from_name(video_name)
                
                # 处理人数 (虽然你说暂时没有，但我尝试从文本里帮你自动提取了)
                person_count = extract_person_count(description)

                # 执行插入
                cursor.execute(sql, (
                    video_name,
                    video_path,
                    person_count,
                    description,
                    created_time
                ))
                success_count += 1
                print(f"正在导入: {video_name} | 时间: {created_time} | 人数: {person_count}")

            # 3. 提交事务
            conn.commit()
            print(f"\n🎉 导入完成！成功插入 {success_count} 条数据。")

    except Exception as e:
        print(f"发生错误: {e}")
        conn.rollback() # 发生错误回滚
    finally:
        conn.close()

if __name__ == '__main__':
    main()