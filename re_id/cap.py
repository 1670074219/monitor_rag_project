import cv2
import os
from ultralytics import YOLO

# ----------------配置区域----------------
# 视频路径
video_path = "./camera3_20250624_123553.mp4" 

# 截图保存的文件夹名称
save_dir = "captured_persons_3553"

# 采样频率：每隔几帧保存一次？(设为 1 表示每帧都存，设为 10 表示每 10 帧存一次，防止图片太多)
sample_interval = 5 

# 是否显示实时画面窗口 (True/False)
show_window = False
# ---------------------------------------

def save_person_crops():
    # 1. 准备保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📁 已创建文件夹: {save_dir}")
    else:
        print(f"📁 图片将保存到: {save_dir}")

    # 2. 加载 YOLO 模型 (只用于检测)
    print("正在加载 YOLOv8 模型...")
    model = YOLO('yolov8n.pt')

    # 3. 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return

    frame_count = 0
    saved_count = 0

    print("🚀 开始截取... 按 'q' 键可以提前停止")

    while True:
        ret, frame = cap.read()
        if not ret:
            break # 视频结束

        frame_count += 1

        # 跳帧处理（只处理符合间隔的帧）
        if frame_count % sample_interval != 0:
            continue

        # YOLO 推理
        results = model(frame, verbose=False)
        
        # 获取检测框
        boxes = results[0].boxes
        
        # 筛选出 label 为 0 (person) 的框
        person_boxes = boxes[boxes.cls == 0]

        # 遍历当前帧里的每一个人
        for i, box in enumerate(person_boxes):
            # 获取坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 边界安全检查 (防止坐标超出图片范围报错)
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # ✂️ 核心步骤：截取图片
            crop_img = frame[y1:y2, x1:x2]

            # 如果截出来的图太小（比如误检的噪点），就丢弃
            if crop_img.shape[0] < 20 or crop_img.shape[1] < 20:
                continue

            # 生成文件名: 帧号_序号.jpg
            file_name = f"frame_{frame_count:04d}_p{i}.jpg"
            save_path = os.path.join(save_dir, file_name)

            # 保存图片
            cv2.imwrite(save_path, crop_img)
            saved_count += 1

            # (可选) 在原图上画框，方便观看
            if show_window:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {i}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示进度
        if frame_count % 50 == 0:
            print(f"⏳ 已处理 {frame_count} 帧，累计保存 {saved_count} 张截图...")

        # 显示实时画面
        if show_window:
            cv2.imshow('Monitoring - Press Q to Exit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ 处理完成！")
    print(f"📊 共保存了 {saved_count} 张行人截图。")
    print(f"👉 请打开文件夹查看: ./{save_dir}")

if __name__ == "__main__":
    save_person_crops()