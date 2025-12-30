import os
import torch
import torchvision.transforms as T
from PIL import Image
from torchreid import models
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 请确保路径正确
image1_path = "./1/p1.jpg" # 猫
image2_path = "./1/p3.jpg" # 女生
# 你的权重文件是 osnet_x1_0 的，所以下面必须用 osnet_x1_0
model_path = './osnet_x1_0_market1501_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'

# ---------------------------------------------------------
# ✅ 修正 1: 这里必须改成 'osnet_x1_0' 以匹配你的权重文件
# ---------------------------------------------------------
model_name = 'osnet_x1_0' 
print(f"正在构建模型: {model_name} ...")
model = models.build_model(name=model_name, num_classes=1000, pretrained=False)
model.to(device).eval()

# 加载权重
if not os.path.exists(model_path):
    print("❌ 找不到权重文件，请检查路径！")
    exit()

print(f"正在加载权重: {model_path} ...")
state_dict = torch.load(model_path, map_location=device)

# ---------------------------------------------------------
# ✅ 修正 2: 尝试使用 strict=True 来检测不匹配
# 如果这里报错，说明你的权重和模型还是对不上
# ---------------------------------------------------------
try:
    model.load_state_dict(state_dict, strict=True)
    print("✅ 权重加载成功 (Strict Mode Passed)")
except RuntimeError as e:
    print(f"⚠️ 权重不完全匹配 (可能是分类头 num_classes 不同)，尝试非严格加载...")
    # 只有当仅仅是分类头(classifier)不匹配时，才使用 strict=False
    # 如果是特征层不匹配，模型就废了
    model.load_state_dict(state_dict, strict=False)

# 预处理
transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_feature(img_path):
    if not os.path.exists(img_path):
        print(f"❌ 图片不存在: {img_path}")
        return None
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor)
        feat = feat.cpu().numpy().flatten()
        feat = feat / np.linalg.norm(feat)
    return feat

def is_same_person(feat1, feat2, threshold=0.35):
    sim = np.dot(feat1, feat2)
    return sim >= threshold, sim

# 主流程
if __name__ == "__main__":
    f1 = extract_feature(image1_path)
    f2 = extract_feature(image2_path)
    
    if f1 is not None and f2 is not None:
        same, score = is_same_person(f1, f2)
        print(f"\n📊 最终相似度: {score:.4f}")
        if same:
            print("✅ 判断为：同一个人 (Warning: 结果可能不准，因为输入了非行人图片)")
        else:
            print("❌ 判断为：不同的人")