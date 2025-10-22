import os
import json
import cv2
import base64
import glob
import numpy as np
from config import *

def save_json(file_name,content):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=4, ensure_ascii=False)

def save_txt(file_name,content):
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(content)

def load_json(file_name,):
    with open(file_name, "r", encoding="utf-8") as f:
        return json.load(f)

def load_txt(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        return f.read()

def parse_json_response(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("⚠️ 模型输出不是合法的 JSON 格式：")
        print(response)
        return None

def resize_image(image, size=448):
    """等比例缩放图像"""
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def get_images(directory, step=8, size=448):
    """提取目录中的图像或视频帧"""
    results = []
    image_files = []
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif')
    video_extensions = (".mp4", ".avi")

    if os.path.isfile(directory) and directory.lower().endswith(video_extensions):
        cap = cv2.VideoCapture(directory)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            frame = resize_image(frame, size)  # 等比例缩放
            _, buffer = cv2.imencode(".jpg", frame)
            base64_image = base64.b64encode(buffer).decode("utf-8")
            results.append(f"data:image/jpeg;base64,{base64_image}")

        cap.release()
        return results

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    if len(image_files) > 1200:
        return None
    image_files.sort()
    image_files = image_files[::step]  # 每隔step张取一张

    for file in image_files:
        image = cv2.imread(file)
        if image is None:
            continue  # 跳过无法读取的图片
        image = resize_image(image, size)  # 等比例缩放
        result = model.predict(source=image, classes=[0, 1, 2, 3, 5, 7], conf=0.5)
        if result[0].boxes and len(result[0].boxes.data) > 0:
            _, buffer = cv2.imencode(".jpg", image)
            base64_image = base64.b64encode(buffer).decode("utf-8")
            results.append(f"data:image/jpeg;base64,{base64_image}")

    return results

def llm_inference(query, system_prompt, history=None):
    completion = client.chat.completions.create(
        model=llm_model_name,
        temperature=temperature,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query}],
    )
    return completion.choices[0].message.content

def vlm_inference(selected_images, system_prompt):
    completion = client.chat.completions.create(
        model=vlm_model_name,
        temperature=temperature,
        messages=[
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "video", "video": selected_images},
            ]},
        ],
    )
    return completion.choices[0].message.content

def rag_embedding(logs):
    completion = client.embeddings.create(
        model=emb_model_name,
        input=logs,
        dimensions=emb_d,
        encoding_format="float"
    )
    resp = completion.model_dump_json()
    return [data["embedding"] for data in parse_json_response(resp)["data"]]
