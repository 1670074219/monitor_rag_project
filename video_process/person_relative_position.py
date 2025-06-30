import cv2
import numpy as np
import torch
from ultralytics import YOLO
import json
from datetime import datetime
import os

# DeepSORT imports
from deepsort.deepsort import DeepSort
from deepsort.deep.feature_extractor import Extractor

class PersonRelativePositionTracker:
    """
    基于四个参考点的行人相对位置跟踪器
    
    功能：
    1. 定义四个参考坐标点
    2. 跟踪视频中的行人
    3. 计算行人相对于参考点的像素坐标
    4. 返回实时的位置信息
    """
    
    def __init__(self, reference_points, yolo_model_path='video_process/yolo/yolo11s.pt', 
                 deepsort_model_path='video_process/deepsort/deep/checkpoint/ckpt.t7'):
        """
        初始化跟踪器
        
        :param reference_points: 四个参考点的像素坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        :param yolo_model_path: YOLO模型路径
        :param deepsort_model_path: DeepSORT特征提取器模型路径
        """
        self.reference_points = reference_points
        self.validate_reference_points()
        
        # 计算参考区域的边界
        self.reference_bounds = self.calculate_reference_bounds()
        
        # 初始化模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_yolo_model(yolo_model_path)
        self.init_deepsort(deepsort_model_path)
        
        # 存储跟踪结果
        self.tracking_results = {}
        self.frame_count = 0
        
    def validate_reference_points(self):
        """验证参考点的有效性"""
        if len(self.reference_points) != 4:
            raise ValueError("必须提供恰好4个参考点")
        
        for i, point in enumerate(self.reference_points):
            if len(point) != 2:
                raise ValueError(f"参考点{i+1}格式错误，应为(x, y)格式")
            if not all(isinstance(coord, (int, float)) for coord in point):
                raise ValueError(f"参考点{i+1}坐标必须为数字")
    
    def calculate_reference_bounds(self):
        """计算参考点形成的边界框"""
        x_coords = [point[0] for point in self.reference_points]
        y_coords = [point[1] for point in self.reference_points]
        
        return {
            'min_x': min(x_coords),
            'max_x': max(x_coords),
            'min_y': min(y_coords),
            'max_y': max(y_coords),
            'center_x': sum(x_coords) / 4,
            'center_y': sum(y_coords) / 4
        }
    
    def init_yolo_model(self, model_path):
        """初始化YOLO模型"""
        try:
            self.yolo_model = YOLO(model_path)
            self.yolo_model.to(self.device)
            print(f"✅ YOLO模型加载成功: {model_path}")
        except Exception as e:
            print(f"❌ YOLO模型加载失败: {e}")
            raise
    
    def init_deepsort(self, model_path):
        """初始化DeepSORT跟踪器"""
        try:
            self.feature_extractor = Extractor(model_path, use_cuda=torch.cuda.is_available())
            self.deepsort = DeepSort(
                max_dist=0.2, min_confidence=0.7, nms_max_overlap=0.5,
                max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100
            )
            print(f"✅ DeepSORT跟踪器初始化成功")
        except Exception as e:
            print(f"❌ DeepSORT初始化失败: {e}")
            raise
    
    def calculate_relative_position(self, person_pixel_coord):
        """
        计算行人相对于四个参考点的位置信息
        
        :param person_pixel_coord: 行人的像素坐标 (x, y)
        :return: 相对位置信息字典
        """
        px, py = person_pixel_coord
        
        # 1. 相对于每个参考点的距离和方向
        relative_to_points = []
        for i, (ref_x, ref_y) in enumerate(self.reference_points):
            # 计算距离
            distance = np.sqrt((px - ref_x)**2 + (py - ref_y)**2)
            
            # 计算角度（以参考点为原点）
            angle = np.arctan2(py - ref_y, px - ref_x) * 180 / np.pi
            
            # 计算相对偏移
            offset_x = px - ref_x
            offset_y = py - ref_y
            
            relative_to_points.append({
                'point_index': i + 1,
                'reference_point': (ref_x, ref_y),
                'distance': round(distance, 2),
                'angle_degrees': round(angle, 2),
                'offset_x': round(offset_x, 2),
                'offset_y': round(offset_y, 2)
            })
        
        # 2. 相对于参考区域中心的位置
        center_x = self.reference_bounds['center_x']
        center_y = self.reference_bounds['center_y']
        center_distance = np.sqrt((px - center_x)**2 + (py - center_y)**2)
        center_angle = np.arctan2(py - center_y, px - center_x) * 180 / np.pi
        
        # 3. 在参考区域内的相对位置（百分比）
        bounds = self.reference_bounds
        relative_x = (px - bounds['min_x']) / (bounds['max_x'] - bounds['min_x']) if bounds['max_x'] != bounds['min_x'] else 0
        relative_y = (py - bounds['min_y']) / (bounds['max_y'] - bounds['min_y']) if bounds['max_y'] != bounds['min_y'] else 0
        
        # 4. 是否在参考区域内
        is_inside = (bounds['min_x'] <= px <= bounds['max_x'] and 
                    bounds['min_y'] <= py <= bounds['max_y'])
        
        return {
            'person_pixel_position': (px, py),
            'relative_to_each_point': relative_to_points,
            'relative_to_center': {
                'center_position': (center_x, center_y),
                'distance': round(center_distance, 2),
                'angle_degrees': round(center_angle, 2),
                'offset_x': round(px - center_x, 2),
                'offset_y': round(py - center_y, 2)
            },
            'relative_position_percentage': {
                'x_percent': round(relative_x * 100, 2),
                'y_percent': round(relative_y * 100, 2)
            },
            'is_inside_reference_area': is_inside,
            'reference_bounds': bounds
        }
    
    def process_video(self, video_path, output_file=None, skip_frames=5, max_frames=None):
        """
        处理视频并跟踪行人位置
        
        :param video_path: 视频文件路径
        :param output_file: 输出结果文件路径（JSON格式）
        :param skip_frames: 跳帧数量（提高处理速度）
        :param max_frames: 最大处理帧数
        :return: 跟踪结果字典
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"🎥 开始处理视频: {video_path}")
        print(f"📊 视频信息: {total_frames}帧, {fps}FPS")
        print(f"📍 参考点: {self.reference_points}")
        
        frame_results = {}
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # 跳帧处理
                if self.frame_count % skip_frames != 0:
                    continue
                
                # 最大帧数限制
                if max_frames and self.frame_count > max_frames:
                    break
                
                # 进行人员检测和跟踪
                person_positions = self.detect_and_track_persons(frame)
                
                if person_positions:
                    frame_time = self.frame_count / fps
                    
                    frame_results[self.frame_count] = {
                        'frame_number': self.frame_count,
                        'timestamp_seconds': round(frame_time, 2),
                        'detected_persons': []
                    }
                    
                    for person_data in person_positions:
                        # 计算相对位置
                        relative_pos = self.calculate_relative_position(person_data['center_point'])
                        
                        person_result = {
                            'track_id': person_data['track_id'],
                            'bbox': person_data['bbox'],
                            'center_point': person_data['center_point'],
                            'foot_point': person_data['foot_point'],
                            'relative_position': relative_pos
                        }
                        
                        frame_results[self.frame_count]['detected_persons'].append(person_result)
                    
                    print(f"📍 帧{self.frame_count}: 检测到{len(person_positions)}个行人")
                
                # 显示处理进度
                if self.frame_count % (skip_frames * 10) == 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"🔄 处理进度: {progress:.1f}%")
        
        finally:
            cap.release()
        
        # 保存结果
        result_data = {
            'video_info': {
                'video_path': video_path,
                'total_frames': total_frames,
                'fps': fps,
                'processed_frames': len(frame_results)
            },
            'reference_points': self.reference_points,
            'reference_bounds': self.reference_bounds,
            'tracking_results': frame_results,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            print(f"💾 结果已保存到: {output_file}")
        
        return result_data
    
    def detect_and_track_persons(self, frame):
        """
        在单帧中检测和跟踪行人
        
        :param frame: 输入帧
        :return: 检测到的行人位置列表
        """
        try:
            # YOLO检测
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.yolo_model(img_rgb)
            
            # 提取人员检测结果
            person_detections = []
            for res in results:
                for box in res.boxes:
                    if box.cls == 0:  # 类别0是'person'
                        person_detections.append(
                            box.xyxy[0].tolist() + [box.conf[0].item()] + [box.cls[0].item()]
                        )
            
            if not person_detections:
                return []
            
            person_detections = torch.tensor(person_detections)
            
            # 转换为DeepSORT格式
            bbox_xywh = person_detections[:, :4].clone()
            bbox_xywh[:, 0] = (person_detections[:, 0] + person_detections[:, 2]) / 2
            bbox_xywh[:, 1] = (person_detections[:, 1] + person_detections[:, 3]) / 2
            bbox_xywh[:, 2] = person_detections[:, 2] - person_detections[:, 0]
            bbox_xywh[:, 3] = person_detections[:, 3] - person_detections[:, 1]
            
            confidences = person_detections[:, 4]
            
            # DeepSORT跟踪
            outputs = self.deepsort.update(bbox_xywh.cpu(), confidences.cpu(), frame, self.feature_extractor)
            
            if len(outputs) == 0:
                return []
            
            # 构建结果
            person_positions = []
            for output in outputs:
                x1, y1, x2, y2, track_id = output
                
                # 计算中心点和脚点
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                foot_x = center_x
                foot_y = int(y2)  # 脚点在边界框底部
                
                person_positions.append({
                    'track_id': int(track_id),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'center_point': (center_x, center_y),
                    'foot_point': (foot_x, foot_y)
                })
            
            return person_positions
            
        except Exception as e:
            print(f"❌ 检测跟踪错误: {e}")
            return []
    
    def visualize_results(self, frame, person_positions, show_reference_points=True):
        """
        在帧上可视化跟踪结果
        
        :param frame: 输入帧
        :param person_positions: 行人位置列表
        :param show_reference_points: 是否显示参考点
        :return: 可视化后的帧
        """
        vis_frame = frame.copy()
        
        # 绘制参考点
        if show_reference_points:
            for i, (x, y) in enumerate(self.reference_points):
                cv2.circle(vis_frame, (int(x), int(y)), 8, (0, 255, 255), -1)
                cv2.putText(vis_frame, f'P{i+1}', (int(x)+10, int(y)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 绘制参考区域边界
            bounds = self.reference_bounds
            cv2.rectangle(vis_frame, 
                         (int(bounds['min_x']), int(bounds['min_y'])),
                         (int(bounds['max_x']), int(bounds['max_y'])),
                         (0, 255, 255), 2)
        
        # 绘制行人检测结果
        for person in person_positions:
            x1, y1, x2, y2 = person['bbox']
            track_id = person['track_id']
            center_x, center_y = person['center_point']
            foot_x, foot_y = person['foot_point']
            
            # 边界框
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 跟踪ID
            cv2.putText(vis_frame, f'ID:{track_id}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 中心点
            cv2.circle(vis_frame, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # 脚点
            cv2.circle(vis_frame, (foot_x, foot_y), 5, (0, 0, 255), -1)
        
        return vis_frame


def main():
    """
    使用示例
    """
    # 定义四个参考点（示例坐标）
    reference_points = [
        (479, 117),   # 点1：左上
        (629, 122),   # 点2：右上
        (1033, 717),  # 点3：右下
        (206, 716)    # 点4：左下
    ]
    
    # 创建跟踪器
    tracker = PersonRelativePositionTracker(reference_points)
    
    # 处理视频
    video_path = 'video_process/saved_video/camera6_20250617_153316.mp4'
    output_file = 'video_process/person_relative_positions.json'
    
    try:
        results = tracker.process_video(
            video_path=video_path,
            output_file=output_file,
            skip_frames=5,  # 每5帧处理一次
            max_frames=1000  # 最多处理1000帧
        )
        
        print("🎉 处理完成！")
        print(f"📊 总共处理了 {len(results['tracking_results'])} 帧")
        
        # 输出一些示例结果
        for frame_num, frame_data in list(results['tracking_results'].items())[:3]:
            print(f"\n📍 帧 {frame_num} 结果示例:")
            for person in frame_data['detected_persons']:
                rel_pos = person['relative_position']
                print(f"  👤 行人ID {person['track_id']}:")
                print(f"     像素位置: {rel_pos['person_pixel_position']}")
                print(f"     相对中心距离: {rel_pos['relative_to_center']['distance']}像素")
                print(f"     区域内位置: {rel_pos['relative_position_percentage']}%")
                print(f"     是否在参考区域内: {rel_pos['is_inside_reference_area']}")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")


if __name__ == '__main__':
    main() 