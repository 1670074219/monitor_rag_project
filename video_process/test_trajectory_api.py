#!/usr/bin/env python3
"""
轨迹API测试脚本
"""

import requests
import json

def test_trajectory_api():
    """测试轨迹API接口"""
    
    base_url = "http://localhost:5000"
    
    print("🚀 测试轨迹API接口")
    print("=" * 50)
    
    # 1. 测试获取3D事件数据（包含轨迹标志）
    print("\n1. 测试获取3D事件数据...")
    try:
        response = requests.get(f"{base_url}/api/events_3d")
        if response.status_code == 200:
            events = response.json()
            print(f"   ✅ 成功获取 {len(events)} 个事件")
            
            # 查找有轨迹数据的事件
            events_with_trajectory = [e for e in events if e.get('has_trajectory')]
            print(f"   📊 有轨迹数据的事件: {len(events_with_trajectory)} 个")
            
            if events_with_trajectory:
                test_event = events_with_trajectory[0]
                test_event_id = test_event['id']
                print(f"   🎯 选择测试事件: {test_event_id}")
                return test_event_id
            else:
                print("   ❌ 没有找到有轨迹数据的事件")
                return None
        else:
            print(f"   ❌ 请求失败: {response.status_code}")
            return None
    except Exception as e:
        print(f"   ❌ 请求异常: {e}")
        return None

def test_event_trajectory(event_id):
    """测试特定事件的轨迹数据"""
    
    base_url = "http://localhost:5000"
    
    print(f"\n2. 测试事件 {event_id} 的轨迹数据...")
    
    # 测试原始轨迹数据
    try:
        response = requests.get(f"{base_url}/api/trajectory/{event_id}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ 原始轨迹数据获取成功")
            print(f"   📊 人数: {data.get('person_count', 0)}")
            print(f"   📊 轨迹数量: {len(data.get('trajectories', []))}")
            
            for i, traj in enumerate(data.get('trajectories', [])):
                print(f"      轨迹 {i+1}: Track ID {traj.get('track_id')}, {traj.get('trajectory_length')} 个点")
        else:
            print(f"   ❌ 原始轨迹数据请求失败: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 原始轨迹数据请求异常: {e}")
    
    # 测试场景坐标轨迹数据
    try:
        response = requests.get(f"{base_url}/api/trajectory/{event_id}/scene_coords")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ 场景坐标轨迹数据获取成功")
            print(f"   📊 人数: {data.get('person_count', 0)}")
            print(f"   📊 轨迹数量: {len(data.get('trajectories', []))}")
            
            for i, traj in enumerate(data.get('trajectories', [])):
                track_id = traj.get('track_id')
                color = traj.get('color')
                coords = traj.get('coordinates', [])
                print(f"      轨迹 {i+1}: Track ID {track_id}, 颜色 {color}, {len(coords)} 个点")
                
                if coords:
                    start_point = coords[0]
                    end_point = coords[-1]
                    print(f"         起点: ({start_point['x']:.3f}, {start_point['y']:.3f}, {start_point['z']:.3f})")
                    print(f"         终点: ({end_point['x']:.3f}, {end_point['y']:.3f}, {end_point['z']:.3f})")
        else:
            print(f"   ❌ 场景坐标轨迹数据请求失败: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 场景坐标轨迹数据请求异常: {e}")

def main():
    """主函数"""
    
    print("🧪 轨迹API功能测试")
    print("请确保API服务器运行在 http://localhost:5000")
    print("=" * 60)
    
    # 测试获取事件列表
    test_event_id = test_trajectory_api()
    
    if test_event_id:
        # 测试特定事件的轨迹数据
        test_event_trajectory(test_event_id)
    
    print("\n" + "=" * 60)
    print("📋 测试完成！")

if __name__ == "__main__":
    main() 