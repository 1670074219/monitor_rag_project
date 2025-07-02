#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON数据迁移到MySQL数据库脚本
支持大文件分批处理，避免内存溢出
"""

import json
import mysql.connector
from mysql.connector import Error
import sys
import os
from typing import Iterator, Dict, Any
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    'host': '219.216.99.151',
    'port': 3306,
    'database': 'monitor_rag',
    'user': 'root',
    'password': 'q1w2e3az',
    'charset': 'utf8mb4',
    'autocommit': False
}

# 批处理大小
BATCH_SIZE = 100

class JSONFileProcessor:
    """JSON文件处理器，支持大文件分批读取"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def parse_json_streaming(self) -> Iterator[tuple]:
        """
        流式解析JSON文件，逐个返回记录
        返回格式: (v_name, video_path, analyse_result, is_embedding, idx)
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for v_name, video_info in data.items():
                    yield (
                        v_name,
                        video_info.get('video_path', ''),
                        video_info.get('analyse_result', ''),
                        1 if video_info.get('is_embedding', False) else 0,
                        video_info.get('idx')
                    )
        except Exception as e:
            logger.error(f"解析JSON文件时出错: {e}")
            raise

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.connection = None
    
    def connect(self):
        """连接数据库"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            logger.info("数据库连接成功")
            return True
        except Error as e:
            logger.error(f"数据库连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开数据库连接"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("数据库连接已关闭")
    
    def create_table_if_not_exists(self):
        """如果表不存在则创建表"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS v_dsp (
            id INT AUTO_INCREMENT PRIMARY KEY,
            v_name VARCHAR(50) NOT NULL,
            video_path VARCHAR(200) NOT NULL,
            analyse_result TEXT,
            is_embedding TINYINT(1) DEFAULT 0,
            idx INT,
            UNIQUE KEY unique_v_name (v_name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(create_table_sql)
            self.connection.commit()
            cursor.close()
            logger.info("表检查/创建完成")
            return True
        except Error as e:
            logger.error(f"创建表失败: {e}")
            return False
    
    def check_existing_records(self):
        """检查现有记录数量"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM v_dsp")
            count = cursor.fetchone()[0]
            cursor.close()
            logger.info(f"数据库中现有记录数: {count}")
            return count
        except Error as e:
            logger.error(f"检查现有记录失败: {e}")
            return 0
    
    def insert_batch(self, records: list) -> int:
        """批量插入记录"""
        insert_sql = """
        INSERT INTO v_dsp (v_name, video_path, analyse_result, is_embedding, idx)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        video_path = VALUES(video_path),
        analyse_result = VALUES(analyse_result),
        is_embedding = VALUES(is_embedding),
        idx = VALUES(idx)
        """
        
        try:
            cursor = self.connection.cursor()
            cursor.executemany(insert_sql, records)
            affected_rows = cursor.rowcount
            self.connection.commit()
            cursor.close()
            return affected_rows
        except Error as e:
            logger.error(f"批量插入失败: {e}")
            self.connection.rollback()
            return 0

def migrate_data(json_file_path: str):
    """主迁移函数"""
    
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        logger.error(f"JSON文件不存在: {json_file_path}")
        return False
    
    # 初始化组件
    processor = JSONFileProcessor(json_file_path)
    db_manager = DatabaseManager(DB_CONFIG)
    
    # 连接数据库
    if not db_manager.connect():
        return False
    
    try:
        # 创建表
        if not db_manager.create_table_if_not_exists():
            return False
        
        # 检查现有记录
        existing_count = db_manager.check_existing_records()
        
        # 开始迁移
        logger.info(f"开始迁移数据，批处理大小: {BATCH_SIZE}")
        
        batch = []
        total_processed = 0
        total_inserted = 0
        
        for record in processor.parse_json_streaming():
            batch.append(record)
            
            if len(batch) >= BATCH_SIZE:
                # 处理当前批次
                inserted = db_manager.insert_batch(batch)
                total_processed += len(batch)
                total_inserted += inserted
                
                logger.info(f"已处理 {total_processed} 条记录，插入/更新 {total_inserted} 条")
                batch = []
        
        # 处理剩余记录
        if batch:
            inserted = db_manager.insert_batch(batch)
            total_processed += len(batch)
            total_inserted += inserted
        
        logger.info(f"迁移完成！总共处理 {total_processed} 条记录，插入/更新 {total_inserted} 条")
        
        # 检查最终记录数
        final_count = db_manager.check_existing_records()
        logger.info(f"迁移后数据库记录数: {final_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"迁移过程中出错: {e}")
        return False
    
    finally:
        db_manager.disconnect()

if __name__ == "__main__":
    json_file_path = "video_process/video_description_back.json"
    
    print("=== JSON数据迁移到MySQL ===")
    print(f"源文件: {json_file_path}")
    print(f"目标数据库: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    print(f"目标表: v_dsp")
    print("=" * 50)
    
    # 确认是否继续
    confirm = input("确认开始迁移？(y/N): ").strip().lower()
    if confirm != 'y':
        print("迁移已取消")
        sys.exit(0)
    
    # 开始迁移
    success = migrate_data(json_file_path)
    
    if success:
        print("\n✅ 迁移成功完成！")
        sys.exit(0)
    else:
        print("\n❌ 迁移失败，请查看日志")
        sys.exit(1) 