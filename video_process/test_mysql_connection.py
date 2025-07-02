#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试MySQL数据库连接和基本功能
"""

import mysql.connector
from mysql.connector import Error
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    'host': '219.216.99.151',
    'port': 3306,
    'database': 'monitor_rag',
    'user': 'root',
    'password': 'q1w2e3az',
    'charset': 'utf8mb4'
}

def test_database_connection():
    """测试数据库连接"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            db_info = connection.get_server_info()
            logger.info(f"✅ 成功连接到MySQL服务器，版本: {db_info}")
            
            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE();")
            database_name = cursor.fetchone()
            logger.info(f"✅ 当前数据库: {database_name[0]}")
            
            return connection
        
    except Error as e:
        logger.error(f"❌ 数据库连接失败: {e}")
        return None

def test_table_structure(connection):
    """测试表结构"""
    try:
        cursor = connection.cursor()
        
        # 检查表是否存在
        cursor.execute("SHOW TABLES LIKE 'v_dsp'")
        result = cursor.fetchone()
        
        if result:
            logger.info("✅ 表 v_dsp 存在")
            
            # 查看表结构
            cursor.execute("DESCRIBE v_dsp")
            columns = cursor.fetchall()
            
            logger.info("📋 表结构:")
            for column in columns:
                logger.info(f"  - {column[0]}: {column[1]} (NULL: {column[2]}, Key: {column[3]}, Default: {column[4]})")
                
        else:
            logger.warning("⚠️ 表 v_dsp 不存在")
            
        cursor.close()
        
    except Error as e:
        logger.error(f"❌ 检查表结构失败: {e}")

def test_data_statistics(connection):
    """测试数据统计"""
    try:
        cursor = connection.cursor()
        
        # 统计总记录数
        cursor.execute("SELECT COUNT(*) FROM v_dsp")
        total_count = cursor.fetchone()[0]
        logger.info(f"📊 总记录数: {total_count}")
        
        # 统计已分析的记录数
        cursor.execute("SELECT COUNT(*) FROM v_dsp WHERE analyse_result IS NOT NULL AND analyse_result != ''")
        analyzed_count = cursor.fetchone()[0]
        logger.info(f"📊 已分析记录数: {analyzed_count}")
        
        # 统计未分析的记录数
        cursor.execute("SELECT COUNT(*) FROM v_dsp WHERE analyse_result IS NULL OR analyse_result = ''")
        unanalyzed_count = cursor.fetchone()[0]
        logger.info(f"📊 未分析记录数: {unanalyzed_count}")
        
        # 显示进度
        if total_count > 0:
            progress = (analyzed_count / total_count) * 100
            logger.info(f"📈 分析进度: {progress:.2f}%")
            
        # 查看前5条未分析的记录
        cursor.execute("""
            SELECT v_name, video_path 
            FROM v_dsp 
            WHERE analyse_result IS NULL OR analyse_result = '' 
            LIMIT 5
        """)
        unanalyzed_videos = cursor.fetchall()
        
        if unanalyzed_videos:
            logger.info("🎬 前5个未分析的视频:")
            for i, (v_name, video_path) in enumerate(unanalyzed_videos, 1):
                logger.info(f"  {i}. {v_name}: {video_path}")
        else:
            logger.info("🎉 所有视频都已分析完成！")
            
        cursor.close()
        
    except Error as e:
        logger.error(f"❌ 获取统计信息失败: {e}")

def test_sample_queries(connection):
    """测试示例查询"""
    try:
        cursor = connection.cursor(dictionary=True)
        
        # 查询一条已分析的记录示例
        cursor.execute("""
            SELECT v_name, video_path, analyse_result 
            FROM v_dsp 
            WHERE analyse_result IS NOT NULL AND analyse_result != '' 
            LIMIT 1
        """)
        sample_analyzed = cursor.fetchone()
        
        if sample_analyzed:
            logger.info("✅ 已分析记录示例:")
            logger.info(f"  视频名称: {sample_analyzed['v_name']}")
            logger.info(f"  视频路径: {sample_analyzed['video_path']}")
            logger.info(f"  分析结果长度: {len(sample_analyzed['analyse_result'])} 字符")
            
        # 查询一条未分析的记录示例
        cursor.execute("""
            SELECT v_name, video_path 
            FROM v_dsp 
            WHERE analyse_result IS NULL OR analyse_result = '' 
            LIMIT 1
        """)
        sample_unanalyzed = cursor.fetchone()
        
        if sample_unanalyzed:
            logger.info("⏳ 未分析记录示例:")
            logger.info(f"  视频名称: {sample_unanalyzed['v_name']}")
            logger.info(f"  视频路径: {sample_unanalyzed['video_path']}")
            
        cursor.close()
        
    except Error as e:
        logger.error(f"❌ 示例查询失败: {e}")

def main():
    """主函数"""
    logger.info("🚀 开始测试MySQL数据库连接...")
    logger.info("=" * 60)
    
    # 测试连接
    connection = test_database_connection()
    if not connection:
        return False
    
    try:
        # 测试表结构
        logger.info("\n" + "=" * 60)
        logger.info("🔍 测试表结构...")
        test_table_structure(connection)
        
        # 测试数据统计
        logger.info("\n" + "=" * 60)
        logger.info("📊 测试数据统计...")
        test_data_statistics(connection)
        
        # 测试示例查询
        logger.info("\n" + "=" * 60)
        logger.info("🔎 测试示例查询...")
        test_sample_queries(connection)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ 所有测试完成！数据库连接正常。")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试过程中发生错误: {e}")
        return False
        
    finally:
        if connection and connection.is_connected():
            connection.close()
            logger.info("🔌 数据库连接已关闭")

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 