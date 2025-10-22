#!/usr/bin/env python3
"""
调试全局轨迹显示问题
检查相关事件是否真的有轨迹数据，以及数据的具体内容
"""

import requests
import json
import sys

def debug_global_trajectory_issue():
    """调试全局轨迹显示问题"""
    base_url = "http://localhost:5000"
    
    print("🔍 调试全局轨迹显示问题")
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
        
        # 找一个有轨迹的事件作为目标
        target_event = None
        for event in events:
            if event.get('has_trajectory'):
                target_event = event
                break
        
        if not target_event:
            print("❌ 没有找到有轨迹的目标事件")
            return
        
        target_id = target_event['id']
        print(f"\n🎯 第2步：选择目标事件")
        print(f"事件ID: {target_id}")
        print(f"事件内容: {target_event.get('content', '(无内容)')}")
        print(f"有轨迹: {target_event.get('has_trajectory', False)}")
        
        # 2. 搜索相关事件
        print(f"\n🔍 第3步：搜索相关事件")
        params = {
            'date': '2025-06-23',
            'threshold': 0.3,
            'keyword_weight': 0.5,
            'semantic_weight': 0.5,
            'max_results': 15
        }
        
        response = requests.get(f"{base_url}/api/related_events/{target_id}", params=params)
        if response.status_code != 200:
            print(f"❌ 搜索相关事件失败: {response.status_code}")
            return
        
        related_events = response.json()
        print(f"✅ 找到 {len(related_events)} 个相关事件")
        
        if len(related_events) == 0:
            print("⚠️ 没有找到相关事件，无法进行轨迹调试")
            return
        
        # 3. 检查每个相关事件的轨迹数据
        print(f"\n📊 第4步：检查相关事件的轨迹数据")
        print("-" * 50)
        
        valid_trajectory_count = 0
        total_trajectory_points = 0
        
        for i, related in enumerate(related_events[:5]):  # 只检查前5个
            event_id = related['event_id']
            total_score = related['total_score']
            
            print(f"\n{i+1}. 事件ID: {event_id}")
            print(f"   相似度得分: {total_score}")
            print(f"   内容: {related['content'][:50]}...")
            
            # 检查scene_coords轨迹数据
            response = requests.get(f"{base_url}/api/trajectory/{event_id}/scene_coords")
            
            if response.status_code == 200:
                trajectory_data = response.json()
                trajectories = trajectory_data.get('trajectories', [])
                
                print(f"   ✅ 轨迹API调用成功")
                print(f"   轨迹数量: {len(trajectories)}")
                
                if trajectories:
                    valid_trajectory_count += 1
                    for j, traj in enumerate(trajectories):
                        coords = traj.get('coordinates', [])
                        color = traj.get('color', '未知')
                        track_id = traj.get('track_id', '未知')
                        
                        total_trajectory_points += len(coords)
                        
                        print(f"     轨迹 {j+1}: Track ID={track_id}, 颜色={color}, 点数={len(coords)}")
                        
                        if len(coords) > 0:
                            start_point = coords[0]
                            end_point = coords[-1]
                            print(f"     起点: x={start_point.get('x', 'N/A')}, y={start_point.get('y', 'N/A')}, z={start_point.get('z', 'N/A')}")
                            print(f"     终点: x={end_point.get('x', 'N/A')}, y={end_point.get('y', 'N/A')}, z={end_point.get('z', 'N/A')}")
                        else:
                            print(f"     ⚠️ 轨迹没有坐标点")
                else:
                    print(f"   ⚠️ 该事件没有轨迹数据")
            else:
                print(f"   ❌ 轨迹API调用失败: {response.status_code}")
                if response.text:
                    try:
                        error_data = response.json()
                        print(f"   错误: {error_data.get('error', response.text)}")
                    except:
                        print(f"   错误: {response.text}")
        
        # 4. 总结分析
        print(f"\n📋 第5步：总结分析")
        print("=" * 50)
        print(f"总检查事件数: {min(5, len(related_events))}")
        print(f"有有效轨迹的事件数: {valid_trajectory_count}")
        print(f"总轨迹点数: {total_trajectory_points}")
        
        if valid_trajectory_count == 0:
            print("\n❌ 问题分析：所有相关事件都没有轨迹数据")
            print("可能原因：")
            print("1. 事件虽然被标记为有轨迹，但实际轨迹数据为空")
            print("2. 轨迹数据存储格式有问题")
            print("3. 坐标转换过程中出现错误")
        elif total_trajectory_points == 0:
            print("\n❌ 问题分析：事件有轨迹对象但没有坐标点")
            print("可能原因：")
            print("1. 坐标数组为空")
            print("2. 坐标格式不正确")
        else:
            print(f"\n✅ 轨迹数据正常：{valid_trajectory_count}个事件，{total_trajectory_points}个轨迹点")
            print("问题可能在前端3D渲染：")
            print("1. 轨迹线条创建失败")
            print("2. 轨迹在3D场景中不可见（颜色、透明度、位置问题）")
            print("3. 3D场景未正确更新")
            
            # 建议检查前端控制台
            print("\n💡 建议检查：")
            print("1. 打开浏览器开发者工具，查看Console标签页")
            print("2. 点击'全局轨迹'按钮时观察控制台输出")
            print("3. 查看是否有JavaScript错误或警告")
            print("4. 检查3D场景是否正常加载")
        
        print(f"\n🔧 调试建议：")
        if valid_trajectory_count > 0:
            print("轨迹数据正常，问题可能在前端显示逻辑：")
            print("- 检查浏览器控制台是否有JavaScript错误")
            print("- 确认3D场景是否正常初始化")
            print("- 检查轨迹线条的颜色和透明度设置")
            print("- 确认摄像机位置是否能看到轨迹")
        else:
            print("需要检查后端轨迹数据：")
            print("- 检查video_description.json中的轨迹数据格式")
            print("- 确认坐标转换函数convert_pixel_to_scene_coords是否正常")
            print("- 检查原始轨迹数据是否存在")
        
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败，请确保API服务器正在运行 (python api_server.py)")
        return False
    except Exception as e:
        print(f"❌ 调试过程出错: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("全局轨迹显示问题调试工具")
    print("请确保API服务器已启动：python api_server.py")
    print()
    
    debug_global_trajectory_issue() 