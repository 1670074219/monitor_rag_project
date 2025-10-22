from utils import *
from prompts import vlm_prompt

class SurveillanceVLMAnalyzer:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt

    def analyze(self, video_dir):
        selected_images = get_images(video_dir)
        if not selected_images:
            print(f"⚠️ 目录 {video_dir} 无图片或无法处理")
            return None

        try:
            resp = vlm_inference(selected_images, self.system_prompt)
            print(f"🎥 处理完成: {video_dir}\n{resp}")
            parsed = parse_json_response(resp.replace("\n\n","").replace("\n",""))
            return parsed.get("视频描述")

        except Exception as e:
            print(f"❌ 分析 {video_dir} 时出错: {e}")
            return None

    def batch_analyze(self, root_dir, json_save_file):
        results = {}
        for root, dirs, files in os.walk(root_dir):
            if not files:
                continue  # 跳过空目录

            print(f"\n🗂️ 正在处理目录: {root}")
            result = self.analyze(root)
            if result:
                results[root] = result

        save_json(json_save_file, results)
        print(f"\n🚀 批量处理完成，结果保存到 {json_save_file}")

if __name__ == "__main__":
    analyzer = SurveillanceVLMAnalyzer(system_prompt=vlm_prompt)
    analyzer.batch_analyze(test_videos_dir, test_desc_file)
