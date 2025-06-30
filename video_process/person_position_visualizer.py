import cv2
import numpy as np
import json
from person_position_api import PersonPositionAPI

class PersonPositionVisualizer:
    """
    行人位置可视化工具
    在图像上绘制参考点、行人位置和相对位置信息
    """
    
    def __init__(self):
        self.api = PersonPositionAPI()
        self.colors = {
            'reference_points': (0, 255, 255),  # 黄色
            'reference_area': (0, 255, 255),    # 黄色
            'person_bbox': (0, 255, 0),         # 绿色
            'person_center': (255, 0, 0),       # 蓝色
            'person_foot': (0, 0, 255),         # 红色
            'text': (255, 255, 255),            # 白色
            'connection_line': (128, 128, 128)  # 灰色
        }
    
    def draw_reference_points(self, image, reference_points):
        """在图像上绘制参考点和参考区域"""
        # 绘制参考点
        for i, (x, y) in enumerate(reference_points):
            cv2.circle(image, (int(x), int(y)), 12, self.colors['reference_points'], -1)
            cv2.circle(image, (int(x), int(y)), 15, self.colors['reference_points'], 2)
            cv2.putText(image, f'P{i+1}', (int(x)+20, int(y)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['reference_points'], 2)
        
        # 绘制参考区域边界
        x_coords = [point[0] for point in reference_points]
        y_coords = [point[1] for point in reference_points]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)),
                     self.colors['reference_area'], 3)
        
        # 绘制中心点
        center_x = sum(x_coords) / 4
        center_y = sum(y_coords) / 4
        cv2.circle(image, (int(center_x), int(center_y)), 8, self.colors['reference_points'], -1)
        cv2.putText(image, 'CENTER', (int(center_x)+15, int(center_y)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['reference_points'], 2)
        
        return image
    
    def draw_person_info(self, image, person_data):
        """在图像上绘制单个行人的信息"""
        track_id = person_data['person_id']
        pixel_pos = person_data['pixel_position']
        bbox = person_data.get('bbox', None)
        
        px, py = pixel_pos
        
        # 绘制边界框（如果有的话）
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), self.colors['person_bbox'], 2)
        
        # 绘制中心点
        cv2.circle(image, (int(px), int(py)), 8, self.colors['person_center'], -1)
        
        # 绘制脚点（假设在中心点下方）
        foot_y = bbox[3] if bbox else py + 20
        cv2.circle(image, (int(px), int(foot_y)), 6, self.colors['person_foot'], -1)
        
        # 绘制到最近参考点的连线
        closest_point = person_data['closest_reference_point']
        if closest_point:
            ref_points = self.api.tracker.reference_points
            ref_point = ref_points[closest_point['point_index'] - 1]
            cv2.line(image, (int(px), int(py)), (int(ref_point[0]), int(ref_point[1])),
                    self.colors['connection_line'], 2)
        
        # 添加文本信息
        info_lines = [
            f"ID: {track_id}",
            f"位置: ({px}, {py})",
            f"最近: P{closest_point['point_index']} ({closest_point['distance']:.0f}px)",
            f"区域: {person_data['position_percentage']['x_percent']:.0f}%, {person_data['position_percentage']['y_percent']:.0f}%",
            f"区域内: {'是' if person_data['is_inside_area'] else '否'}"
        ]
        
        # 确定文本位置（避免超出图像边界）
        text_x = max(10, min(px + 20, image.shape[1] - 200))
        text_y = max(30, py - 80)
        
        # 绘制文本背景
        text_bg_height = len(info_lines) * 25 + 10
        cv2.rectangle(image, (text_x - 5, text_y - 25), 
                     (text_x + 220, text_y + text_bg_height - 25),
                     (0, 0, 0), -1)
        cv2.rectangle(image, (text_x - 5, text_y - 25), 
                     (text_x + 220, text_y + text_bg_height - 25),
                     self.colors['text'], 1)
        
        # 绘制文本
        for i, line in enumerate(info_lines):
            cv2.putText(image, line, (text_x, text_y + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        return image
    
    def visualize_frame(self, image_path, reference_points, output_path=None, show_image=True):
        """
        可视化单帧图像中的行人位置
        
        :param image_path: 输入图像路径
        :param reference_points: 四个参考点坐标
        :param output_path: 输出图像路径（可选）
        :param show_image: 是否显示图像
        :return: 可视化结果
        """
        # 设置参考点
        self.api.set_reference_points(reference_points)
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return {"status": "error", "message": f"无法读取图像: {image_path}"}
        
        # 创建可视化图像副本
        vis_image = image.copy()
        
        # 绘制参考点和参考区域
        vis_image = self.draw_reference_points(vis_image, reference_points)
        
        # 获取行人位置信息
        result = self.api.get_simple_position_info(image)
        
        if result["status"] != "success":
            return result
        
        # 绘制每个行人的信息
        for person_data in result["persons"]:
            vis_image = self.draw_person_info(vis_image, person_data)
        
        # 添加标题信息
        title = f"检测到 {len(result['persons'])} 个行人"
        cv2.putText(vis_image, title, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['text'], 2)
        
        # 保存图像
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"💾 可视化结果已保存到: {output_path}")
        
        # 显示图像
        if show_image:
            cv2.imshow('Person Position Visualization', vis_image)
            print("🖼️  按任意键关闭图像窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return {
            "status": "success",
            "message": f"成功可视化 {len(result['persons'])} 个行人",
            "visualization_image": vis_image,
            "detection_result": result
        }
    
    def create_demo_video(self, video_path, reference_points, output_video_path, max_frames=300):
        """
        创建演示视频，显示实时的行人位置跟踪
        
        :param video_path: 输入视频路径
        :param reference_points: 四个参考点坐标
        :param output_video_path: 输出视频路径
        :param max_frames: 最大处理帧数
        """
        # 设置参考点
        self.api.set_reference_points(reference_points)
        
        # 打开输入视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "error", "message": f"无法打开视频: {video_path}"}
        
        # 获取视频参数
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建输出视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_frames = 0
        
        print(f"🎥 开始创建演示视频...")
        print(f"📊 输入: {video_path}")
        print(f"📊 输出: {output_video_path}")
        
        try:
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 每5帧处理一次（提高处理速度）
                if frame_count % 5 == 0:
                    # 创建可视化帧
                    vis_frame = frame.copy()
                    
                    # 绘制参考点
                    vis_frame = self.draw_reference_points(vis_frame, reference_points)
                    
                    # 检测并绘制行人
                    result = self.api.get_simple_position_info(frame)
                    
                    if result["status"] == "success" and result["persons"]:
                        for person_data in result["persons"]:
                            vis_frame = self.draw_person_info(vis_frame, person_data)
                        
                        # 添加帧信息
                        info_text = f"帧: {frame_count} | 行人: {len(result['persons'])}"
                    else:
                        info_text = f"帧: {frame_count} | 行人: 0"
                    
                    cv2.putText(vis_frame, info_text, (10, height - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
                    
                    # 写入输出视频
                    out.write(vis_frame)
                    processed_frames += 1
                    
                    # 显示进度
                    if processed_frames % 20 == 0:
                        progress = (frame_count / max_frames) * 100
                        print(f"🔄 处理进度: {progress:.1f}% (帧 {frame_count}/{max_frames})")
                else:
                    # 直接写入原始帧
                    out.write(frame)
        
        finally:
            cap.release()
            out.release()
        
        print(f"✅ 演示视频创建完成!")
        print(f"📊 处理了 {processed_frames} 帧，总计 {frame_count} 帧")
        
        return {
            "status": "success",
            "message": f"演示视频创建完成: {output_video_path}",
            "processed_frames": processed_frames,
            "total_frames": frame_count
        }


def demo_visualization():
    """
    可视化演示
    """
    print("🎨 行人位置可视化演示")
    print("=" * 50)
    
    visualizer = PersonPositionVisualizer()
    
    # 定义参考点
    reference_points = [
        (479, 117),   # 左上
        (629, 122),   # 右上
        (1033, 717),  # 右下
        (206, 716)    # 左下
    ]
    
    # 可视化单帧图像
    image_path = "frame.jpg"
    output_image = "person_positions_visualization.jpg"
    
    print(f"🖼️  可视化图像: {image_path}")
    
    result = visualizer.visualize_frame(
        image_path=image_path,
        reference_points=reference_points,
        output_path=output_image,
        show_image=False  # 设为True可以显示图像窗口
    )
    
    if result["status"] == "success":
        print(f"✅ {result['message']}")
        
        # 打印检测结果摘要
        detection_result = result["detection_result"]
        if detection_result["persons"]:
            print(f"\n📊 检测摘要:")
            for i, person in enumerate(detection_result["persons"], 1):
                print(f"   👤 行人{i}: 位置{person['pixel_position']}, "
                      f"最近P{person['closest_reference_point']['point_index']}, "
                      f"区域内: {'是' if person['is_inside_area'] else '否'}")
    else:
        print(f"❌ {result['message']}")


if __name__ == '__main__':
    demo_visualization() 