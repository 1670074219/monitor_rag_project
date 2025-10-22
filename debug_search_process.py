#!/usr/bin/env python3
"""
调试全局轨迹搜索过程，详细展示每一步
"""

import requests
import json
import re

def extract_keywords_simple(text):
    """简单的关键词提取（不需要jieba）"""
    # 定义停用词
    stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
    
    # 使用正则表达式进行简单分词
    words = re.findall(r'[\u4e00-\u9fff]+', text)  # 提取中文字符
    
    # 过滤关键词
    keywords = []
    for word in words:
        if len(word) > 1 and word not in stop_words:
            keywords.append(word)
    
    return keywords

def calculate_keyword_similarity(target_keywords, content):
    """计算关键词相似度"""
    if not target_keywords:
        return 0.0
    
    content_keywords = extract_keywords_simple(content)
    if not content_keywords:
        return 0.0
    
    # 计算交集
    common_keywords = set(target_keywords) & set(content_keywords)
    
    # Jaccard相似度
    union_keywords = set(target_keywords) | set(content_keywords)
    similarity = len(common_keywords) / len(union_keywords) if union_keywords else 0.0
    
    return similarity

def debug_search_process():
    """调试搜索过程"""
    base_url = "http://localhost:5000"
    
    print("🔍 全局轨迹搜索过程调试")
    print("=" * 60)
    
    try:
        # 1. 获取事件列表
        print("📋 第1步：获取事件列表...")
        response = requests.get(f"{base_url}/api/events_3d")
        if response.status_code != 200:
            print(f"❌ 获取事件列表失败: {response.status_code}")
            return
        
        events = response.json()
        print(f"✅ 成功获取 {len(events)} 个事件")
        
        # 找一个有内容的事件作为目标
        target_event = None
        for event in events[:10]:  # 只检查前10个
            if event.get('content') and len(event['content']) > 10:
                target_event = event
                break
        
        if not target_event:
            print("❌ 没有找到适合的目标事件")
            return
        
        target_id = target_event['id']
        target_content = target_event['content']
        
        print(f"\n🎯 第2步：选择目标事件")
        print(f"事件ID: {target_id}")
        print(f"事件内容: {target_content}")
        
        # 2. 提取目标事件关键词
        print(f"\n🔧 第3步：分析目标事件")
        target_keywords = extract_keywords_simple(target_content)
        print(f"提取的关键词: {target_keywords}")
        
        # 3. 手动计算相似度（用于对比）
        print(f"\n📊 第4步：手动计算前5个事件的相似度")
        print("-" * 50)
        
        similar_count = 0
        for i, event in enumerate(events[:10]):
            if event['id'] == target_id:
                continue
                
            content = event.get('content', '')
            if not content:
                continue
            
            # 计算关键词相似度
            keyword_score = calculate_keyword_similarity(target_keywords, content)
            
            # 简单计算总分（只用关键词相似度，因为语义相似度需要模型）
            total_score = keyword_score
            
            print(f"事件 {i+1}: {event['id']}")
            print(f"  内容: {content[:40]}...")
            print(f"  关键词: {extract_keywords_simple(content)}")
            print(f"  关键词相似度: {keyword_score:.3f}")
            print(f"  简单总分: {total_score:.3f}")
            
            if total_score >= 0.1:  # 很低的阈值
                similar_count += 1
                print(f"  ✅ 可能匹配（阈值=0.1）")
            else:
                print(f"  ❌ 不匹配")
            print()
        
        print(f"📈 手动分析结果: {similar_count} 个可能相关的事件")
        
        # 4. 调用API测试不同阈值
        print(f"\n🌐 第5步：测试API搜索（不同阈值）")
        print("-" * 50)
        
        thresholds = [0.1, 0.3, 0.5, 0.6]
        for threshold in thresholds:
            url = f"{base_url}/api/related_events/{target_id}?date=2025-06-23&threshold={threshold}&keyword_weight=0.5&semantic_weight=0.5"
            response = requests.get(url)
            
            if response.status_code == 200:
                related_events = response.json()
                print(f"阈值 {threshold}: 找到 {len(related_events)} 个相关事件")
                
                for j, related in enumerate(related_events[:3]):
                    print(f"  {j+1}. {related['event_id']} (总分: {related['total_score']:.3f})")
            else:
                print(f"阈值 {threshold}: API调用失败 ({response.status_code})")
        
        # 5. 建议
        print(f"\n💡 第6步：搜索建议")
        print("-" * 50)
        print("如果一直显示'没有找到相关事件'，可能的原因:")
        print("1. 阈值太高 → 建议使用 threshold=0.2-0.4")
        print("2. 事件描述太简单 → 选择内容更丰富的事件")
        print("3. jieba库未安装 → 运行: pip install jieba")
        print("4. 语义模型问题 → 增加keyword_weight权重")
        
        print(f"\n推荐设置:")
        print(f"threshold=0.3, keyword_weight=0.6, semantic_weight=0.4")
        
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败，请确保API服务器正在运行")
    except Exception as e:
        print(f"❌ 调试失败: {e}")

if __name__ == "__main__":
    debug_search_process() 