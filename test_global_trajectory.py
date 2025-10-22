#!/usr/bin/env python3
"""
测试全局轨迹API功能
"""

import requests
import json
import sys

def test_related_events_api():
    """测试相关事件搜索API"""
    base_url = "http://localhost:5000"
    
    # 测试获取事件列表
    print("1. 获取3D事件列表...")
    try:
        response = requests.get(f"{base_url}/api/events_3d")
        if response.status_code == 200:
            events = response.json()
            print(f"   成功获取 {len(events)} 个事件")
            
            # 找一个有轨迹的事件进行测试
            event_with_trajectory = None
            for event in events:
                if event.get('has_trajectory'):
                    event_with_trajectory = event
                    break
            
            if event_with_trajectory:
                event_id = event_with_trajectory['id']
                print(f"   选择事件 {event_id} 进行相关事件搜索测试")
                
                # 测试相关事件搜索
                print(f"2. 搜索与事件 {event_id} 相关的事件...")
                response = requests.get(f"{base_url}/api/related_events/{event_id}?date=2025-06-23")
                
                if response.status_code == 200:
                    related_events = response.json()
                    print(f"   找到 {len(related_events)} 个相关事件:")
                    
                    for i, related in enumerate(related_events[:5]):  # 只显示前5个
                        print(f"     {i+1}. 事件ID: {related['event_id']}")
                        print(f"        关键词得分: {related['keyword_score']}")
                        print(f"        语义得分: {related['semantic_score']}")
                        print(f"        总分: {related['total_score']}")
                        print(f"        内容: {related['content'][:50]}...")
                        print()
                    
                    # 测试获取这些相关事件的轨迹数据
                    print("3. 测试获取相关事件的轨迹数据...")
                    trajectory_count = 0
                    for related in related_events[:3]:  # 测试前3个
                        event_id = related['event_id']
                        response = requests.get(f"{base_url}/api/trajectory/{event_id}/scene_coords")
                        if response.status_code == 200:
                            trajectory_data = response.json()
                            if trajectory_data.get('trajectories'):
                                trajectory_count += len(trajectory_data['trajectories'])
                                print(f"   事件 {event_id}: {len(trajectory_data['trajectories'])} 条轨迹")
                    
                    print(f"   总共可显示 {trajectory_count} 条轨迹")
                    
                else:
                    print(f"   相关事件搜索失败: {response.status_code}")
                    print(f"   错误信息: {response.text}")
            else:
                print("   没有找到有轨迹的事件进行测试")
        else:
            print(f"   获取事件列表失败: {response.status_code}")
            print(f"   错误信息: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("   连接失败，请确保API服务器正在运行 (python api_server.py)")
        return False
    except Exception as e:
        print(f"   测试失败: {e}")
        return False
    
    return True

def test_keyword_extraction():
    """测试关键词提取功能"""
    print("4. 测试关键词提取功能...")
    
    # 测试文本
    test_texts = [
        "一个人快速走动，沿着走廊前进，步伐较急",
        "两个人在走廊中间停下交谈，然后分别离开",
        "有人在房间门口徘徊，似乎在等待什么"
    ]
    
    try:
        # 这里我们需要模拟关键词提取
        import jieba
        
        def extract_keywords(text):
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
            words = jieba.cut(text)
            keywords = []
            for word in words:
                word = word.strip()
                if len(word) > 1 and word not in stop_words and word.isalpha():
                    keywords.append(word)
            return keywords
        
        for i, text in enumerate(test_texts):
            keywords = extract_keywords(text)
            print(f"   文本 {i+1}: {text}")
            print(f"   关键词: {', '.join(keywords)}")
            print()
            
    except ImportError:
        print("   jieba库未安装，跳过关键词提取测试")
        print("   请运行: pip install jieba")

def main():
    print("=" * 60)
    print("全局轨迹功能测试")
    print("=" * 60)
    
    # 测试API功能
    if test_related_events_api():
        print("\n✅ API功能测试通过")
    else:
        print("\n❌ API功能测试失败")
        sys.exit(1)
    
    # 测试关键词提取
    test_keyword_extraction()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("\n前端使用说明:")
    print("1. 启动API服务器: python api_server.py")
    print("2. 启动前端: cd frontend && npm run dev")
    print("3. 在前端选择一个有轨迹的事件")
    print("4. 点击'全局轨迹'按钮查看相关事件的轨迹")
    print("=" * 60)

if __name__ == "__main__":
    main() 