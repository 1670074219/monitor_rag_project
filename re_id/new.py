import os
import torch
import torchvision.transforms as T
from PIL import Image
from torchreid import models
import numpy as np

# -----------------------------------------------------------
# ⚙️ 配置区域
# -----------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 图片路径 (修改为你实际的图片路径)
image1_path = "./captured_persons_4553/frame_0110_p0.jpg"  # 示例：穿裙子的女生
image2_path = "./captured_persons_611/frame_0135_p0.jpg"  # 示例：穿格纹的男生

# 2. 权重文件路径 (完全匹配你提供的文件名)
model_path = './osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'

# 3. 模型配置
# ⚠️ 你的权重是 osnet_ain_x1_0，所以这里必须是 ain 版本
model_name = 'osnet_ain_x1_0'
# ⚠️ MSMT17 数据集的类别数通常是 4101
num_classes = 4101 

# -----------------------------------------------------------
# 🛠️ 核心逻辑
# -----------------------------------------------------------

def build_and_load_model():
    print(f"🏗️ 正在构建模型: {model_name} (num_classes={num_classes})...")
    
    # pretrained=False: 禁止程序去国外服务器下载 ImageNet 基础权重，防止卡死
    model = models.build_model(name=model_name, num_classes=num_classes, pretrained=False)
    model.to(device).eval()

    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到权重文件 -> {model_path}")
        print("请确保权重文件在当前目录下，或者修改 model_path 变量。")
        exit(1)

    print(f"📂 正在加载权重...")
    try:
        state_dict = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"❌ 权重文件损坏或无法读取: {e}")
        exit(1)

    # --- 智能处理 'module.' 前缀 (DataParallel 遗留问题) ---
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v # 去掉 'module.'
        else:
            new_state_dict[k] = v
    
    # --- 加载权重 ---
    # strict=False: 允许忽略分类头(classifier)的不匹配，只要特征层(backbone)对得上就行
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    # --- 关键诊断: 检查核心层是否加载 ---
    # 我们只关心 backbone (conv, layer, osnet 等前缀)，不关心 classifier
    real_missing = [k for k in missing_keys if "classifier" not in k and "fc" not in k]
    
    if len(real_missing) > 0:
        print("\n❌❌❌ 严重警告 ❌❌❌")
        print("模型核心层未加载！架构与权重不匹配！")
        print(f"丢失的键示例: {real_missing[:3]}")
        print("这会导致相似度计算完全失效 (结果全是 0.99)。")
        exit(1)
    else:
        print("✅ 权重加载成功！(核心特征层匹配完美)")

    return model

# 初始化模型
model = build_and_load_model()

# 预处理流水线
transform = T.Compose([
    T.Resize((256, 128)), # 强制调整为 ReID 标准尺寸
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_feature(img_path):
    if not os.path.exists(img_path):
        print(f"❌ 图片不存在: {img_path}")
        return None
    
    try:
        img = Image.open(img_path).convert('RGB')
        # 增加一个维度 [C, H, W] -> [1, C, H, W]
        tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat = model(tensor)
            feat = feat.cpu().numpy().flatten()
            # L2 归一化 (对于余弦相似度至关重要)
            norm_val = np.linalg.norm(feat)
            if norm_val > 0:
                feat = feat / norm_val
        return feat
    except Exception as e:
        print(f"处理图片出错 {img_path}: {e}")
        return None

# -----------------------------------------------------------
# ▶️ 主程序
# -----------------------------------------------------------
if __name__ == "__main__":
    print("\n🔎 开始特征对比...")
    
    f1 = extract_feature(image1_path)
    f2 = extract_feature(image2_path)

    if f1 is not None and f2 is not None:
        # 计算余弦相似度
        similarity = np.dot(f1, f2)
        
        print("-" * 30)
        print(f"图片 A: {image1_path}")
        print(f"图片 B: {image2_path}")
        print(f"🔢 相似度得分: {similarity:.4f}")
        print("-" * 30)

        # 判定逻辑
        threshold = 0.35 # 阈值可调，通常 0.3 ~ 0.4
        if similarity > threshold:
            print("✅ 结果: 是同一个人")
            if similarity > 0.95:
                print("⚠️ (如果两张图明显不同，0.95+ 的分数通常意味着模型加载失败，请检查上面的报错)")
        else:
            print("❌ 结果: 不是同一个人")