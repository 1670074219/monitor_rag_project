import cv2
import numpy as np
from person_relative_position import PersonRelativePositionTracker
import json

class PersonPositionAPI:
    """
    简化的行人位置跟踪API
    提供简单易用的接口来跟踪行人相对于参考点的位置
    """
    
    def __init__(self):
        self.tracker = None
        
    def set_reference_points(self, points):
        """
        设置四个参考点
        
        :param points: 四个参考点的像素坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        self.tracker = PersonRelativePositionTracker(points)
        return {"status": "success", "message": f"参考点设置成功: {points}"}
    
    def process_single_frame(self, frame_path_or_array):
        """
        处理单帧图像，返回行人位置信息
        
        :param frame_path_or_array: 图像文件路径或numpy数组
        :return: 行人位置信息
        """
        if self.tracker is None:
            return {"status": "error", "message": "请先设置参考点"}
            
        # 读取图像
        if isinstance(frame_path_or_array, str):
            frame = cv2.imread(frame_path_or_array)
            if frame is None:
                return {"status": "error", "message": f"无法读取图像: {frame_path_or_array}"}
        else:
            frame = frame_path_or_array
            
        # 检测行人
        person_positions = self.tracker.detect_and_track_persons(frame)
        
        if not person_positions:
            return {
                "status": "success", 
                "message": "未检测到行人",
                "persons": []
            }
        
        # 计算相对位置
        result_persons = []
        for person_data in person_positions:
            relative_pos = self.tracker.calculate_relative_position(person_data['center_point'])
            
            result_persons.append({
                'track_id': person_data['track_id'],
                'bbox': person_data['bbox'],
                'center_point': person_data['center_point'],
                'foot_point': person_data['foot_point'],
                'relative_position': relative_pos
            })
        
        return {
            "status": "success",
            "message": f"检测到{len(result_persons)}个行人",
            "persons": result_persons
        }
    
    def get_simple_position_info(self, frame_path_or_array):
        """
        获取简化的位置信息（只返回关键数据）
        
        :param frame_path_or_array: 图像文件路径或numpy数组
        :return: 简化的位置信息
        """
        result = self.process_single_frame(frame_path_or_array)
        
        if result["status"] != "success" or not result["persons"]:
            return result
        
        simple_persons = []
        for person in result["persons"]:
            rel_pos = person['relative_position']
            
            # 找到离哪个参考点最近
            closest_point = min(rel_pos['relative_to_each_point'], 
                               key=lambda x: x['distance'])
            
            simple_info = {
                'person_id': person['track_id'],
                'pixel_position': rel_pos['person_pixel_position'],
                'closest_reference_point': {
                    'point_index': closest_point['point_index'],
                    'distance': closest_point['distance'],
                    'angle': closest_point['angle_degrees']
                },
                'relative_to_center': {
                    'distance': rel_pos['relative_to_center']['distance'],
                    'angle': rel_pos['relative_to_center']['angle_degrees']
                },
                'position_percentage': rel_pos['relative_position_percentage'],
                'is_inside_area': rel_pos['is_inside_reference_area']
            }
            
            simple_persons.append(simple_info)
        
        return {
            "status": "success",
            "message": f"检测到{len(simple_persons)}个行人",
            "persons": simple_persons
        }


def demo_usage():
    """
    使用演示
    """
    print("🎯 行人位置跟踪API演示")
    print("=" * 50)
    
    # 1. 创建API实例
    api = PersonPositionAPI()
    
    # 2. 设置参考点（示例：矩形区域的四个角）
    reference_points = [
        (200, 100),   # 左上角
        (800, 100),   # 右上角  
        (800, 600),   # 右下角
        (200, 600)    # 左下角
    ]
    
    print("📍 设置参考点...")
    setup_result = api.set_reference_points(reference_points)
    print(f"   {setup_result['message']}")
    
    # 3. 处理图像（这里使用frame.jpg作为示例）
    image_path = "frame.jpg"
    
    print(f"\n🔍 分析图像: {image_path}")
    
    # 方法1：获取详细信息
    detailed_result = api.process_single_frame(image_path)
    print(f"   状态: {detailed_result['status']}")
    print(f"   消息: {detailed_result['message']}")
    
    # 方法2：获取简化信息
    simple_result = api.get_simple_position_info(image_path)
    
    if simple_result["status"] == "success" and simple_result["persons"]:
        print(f"\n📊 检测结果:")
        for i, person in enumerate(simple_result["persons"], 1):
            print(f"\n   👤 行人 {i} (ID: {person['person_id']}):")
            print(f"      像素位置: {person['pixel_position']}")
            print(f"      最近参考点: P{person['closest_reference_point']['point_index']} "
                  f"(距离: {person['closest_reference_point']['distance']:.1f}像素)")
            print(f"      相对中心距离: {person['relative_to_center']['distance']:.1f}像素")
            print(f"      区域内位置: X={person['position_percentage']['x_percent']:.1f}%, "
                  f"Y={person['position_percentage']['y_percent']:.1f}%")
            print(f"      是否在参考区域内: {'是' if person['is_inside_area'] else '否'}")
    
    # 4. 保存结果到JSON文件
    output_file = "person_positions_demo.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simple_result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存到: {output_file}")


if __name__ == '__main__':
    demo_usage() 