
# 简单测试示例
from person_position_api import PersonPositionAPI

# 1. 创建API
api = PersonPositionAPI()

# 2. 设置参考点（矩形区域）
points = [
    (100, 100),   # 左上
    (400, 100),   # 右上  
    (400, 300),   # 右下
    (100, 300)    # 左下
]
api.set_reference_points(points)

# 3. 如果有图像文件，可以这样测试：
# result = api.get_simple_position_info("your_image.jpg")
# print(result)

print("✅ 测试代码准备就绪!")
