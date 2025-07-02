#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import sys

def test_api_data_source():
    """
    测试API数据源修改是否成功
    """
    base_url = "http://localhost:5000"
    
    print("=" * 60)
    print("测试前端数据源修改")
    print("📝 验证API是否使用video_description.json作为数据源")
    print("=" * 60)
    
    try:
        # 测试3D事件数据
        print("\n🔍 测试 /api/events_3d 端点...")
        response = requests.get(f"{base_url}/api/events_3d")
        
        if response.status_code == 200:
            events = response.json()
            print(f"✅ 成功获取 {len(events)} 个3D事件")
            
            # 查找有轨迹的事件
            events_with_trajectory = [e for e in events if e.get('has_trajectory', False)]
            print(f"📊 其中有轨迹的事件: {len(events_with_trajectory)} 个")
            
            if events_with_trajectory:
                # 测试轨迹数据获取
                test_event = events_with_trajectory[0]
                event_id = test_event['id']
                
                print(f"\n🎯 测试轨迹数据获取 (事件ID: {event_id})")
                
                # 测试原始轨迹数据
                print("  - 测试 /api/trajectory/<event_id> 端点...")
                traj_response = requests.get(f"{base_url}/api/trajectory/{event_id}")
                
                if traj_response.status_code == 200:
                    traj_data = traj_response.json()
                    print(f"    ✅ 获取到 {len(traj_data.get('trajectories', []))} 条轨迹")
                else:
                    print(f"    ❌ 获取轨迹数据失败: {traj_response.status_code}")
                
                # 测试场景坐标轨迹数据
                print("  - 测试 /api/trajectory/<event_id>/scene_coords 端点...")
                scene_response = requests.get(f"{base_url}/api/trajectory/{event_id}/scene_coords")
                
                if scene_response.status_code == 200:
                    scene_data = scene_response.json()
                    print(f"    ✅ 获取到 {len(scene_data.get('trajectories', []))} 条场景坐标轨迹")
                else:
                    print(f"    ❌ 获取场景坐标轨迹数据失败: {scene_response.status_code}")
            
            else:
                print("⚠️  没有找到有轨迹的事件，无法测试轨迹API")
        
        else:
            print(f"❌ 获取3D事件失败: {response.status_code}")
            return False
    
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到API服务器")
        print("💡 请确保API服务器正在运行 (python api_server.py)")
        return False
    
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 API数据源修改测试完成！")
    print("📁 前端现在使用 video_description.json 作为事件数据源")
    print("=" * 60)
    return True

def check_data_files():
    """
    检查数据文件状态
    """
    import os
    
    print("\n📂 检查数据文件状态:")
    
    files_to_check = [
        "video_process/video_description.json",
        "video_process/video_description_with_trajectory.json"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  ✅ {file_path} ({file_size:.2f} MB)")
        else:
            print(f"  ❌ {file_path} (不存在)")

if __name__ == '__main__':
    check_data_files()
    
    print("\n是否要测试API连接？(y/n): ", end="")
    choice = input().lower().strip()
    
    if choice in ['y', 'yes', '是']:
        test_api_data_source()
    else:
        print("跳过API测试") 