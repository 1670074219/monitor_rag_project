"""
VAD异常检测服务 - 用于监控系统集成
基于LLM的三级异常检测机制
"""

import logging
from typing import Dict, Optional
from VAD.utils import llm_inference, parse_json_response, load_txt, save_txt
from VAD.prompts import llm_prompt1, llm_prompt2, llm_prompt3, rule_merging_prompt
from VAD.config import rules_file

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    视频异常检测器 - 三级检测机制
    
    第一级：基于规则的初步判断
    第二级：事件频率分类（高频/偶发等）
    第三级：偶发事件的深度判断 + 规则自学习
    """
    
    def __init__(self, system_prompt1, system_prompt2, system_prompt3, rule_update_prompt):
        self.system_prompt1 = system_prompt1
        self.system_prompt2 = system_prompt2
        self.system_prompt3 = system_prompt3
        self.rule_merging_prompt = rule_update_prompt
        logger.info("🚀 异常检测器初始化完成")
    
    def detect(self, video_description: str) -> Optional[Dict]:
        """
        对视频描述进行异常检测
        
        Args:
            video_description: 视频描述文本
            
        Returns:
            检测结果字典，包含：
            - is_abnormal: bool, 是否异常
            - detection_result: str, "正常"或"异常"
            - abnormal_events: list, 异常事件列表
            - reason: str, 原因分析
            - confidence: str, 置信度（"高"/"中"/"低"）
            - detection_level: int, 触发的检测级别（1/2/3）
        """
        
        if not video_description or video_description.strip() == "":
            logger.warning("⚠️ 视频描述为空，跳过检测")
            return None
        
        # 跳过已标记为"跳过分析"的视频
        if "跳过分析" in video_description:
            logger.info("📋 视频已被跳过分析，不进行异常检测")
            return {
                "is_abnormal": False,
                "detection_result": "已跳过",
                "abnormal_events": [],
                "reason": video_description,
                "confidence": "N/A",
                "detection_level": 0
            }
        
        try:
            logger.info("🔍 开始第一级异常检测...")
            
            # ========== 第一级：基于规则的异常检测 ==========
            query = f"以下是当前你需要进行异常检测的监控视频描述：\n{video_description}"
            response = llm_inference(query, self.system_prompt1)
            logger.info(f"第一级检测结果:\n{response}")
            
            resp1 = parse_json_response(response)
            if not resp1:
                logger.error("❌ 第一级检测结果解析失败")
                return None
            
            # 如果第一级判定为正常，直接返回
            if resp1.get("异常检测结果") == "正常":
                logger.info("✅ 第一级判定：正常")
                return {
                    "is_abnormal": False,
                    "detection_result": "正常",
                    "abnormal_events": [],
                    "reason": resp1.get("原因分析", "符合正常行为规则"),
                    "confidence": "高",
                    "detection_level": 1
                }
            
            # ========== 第二级：事件频率分类 ==========
            logger.info("🔍 第一级判定为异常，进入第二级检测...")
            abnormal_events = resp1.get("异常事件", [])
            
            if not abnormal_events:
                # 没有具体异常事件，直接返回异常
                logger.warning("⚠️ 检测到异常但无具体事件列表")
                return {
                    "is_abnormal": True,
                    "detection_result": "异常",
                    "abnormal_events": ["未明确的异常行为"],
                    "reason": resp1.get("原因分析", "检测到异常但无具体描述"),
                    "confidence": "中",
                    "detection_level": 1
                }
            
            events_text = "\n".join([f"{i + 1}. {event}" for i, event in enumerate(abnormal_events)])
            response = llm_inference(f"以下为多个行为事件，请分类：\n{events_text}", self.system_prompt2)
            logger.info(f"第二级检测结果:\n{response}")
            
            resp2 = parse_json_response(response)
            if not resp2:
                logger.error("❌ 第二级检测结果解析失败，返回第一级结果")
                return {
                    "is_abnormal": True,
                    "detection_result": "异常",
                    "abnormal_events": abnormal_events,
                    "reason": resp1.get("原因分析", ""),
                    "confidence": "中",
                    "detection_level": 1
                }
            
            # 检查是否存在偶发事件
            has_occasional_event = False
            if isinstance(resp2, list):
                for event in resp2:
                    if event.get("事件类型") == "偶发事件":
                        has_occasional_event = True
                        break
            elif "偶发事件" in str(resp2):
                has_occasional_event = True
            
            # 如果没有偶发事件，直接判定为异常
            if not has_occasional_event:
                logger.info("🚨 第二级判定：异常（非偶发事件）")
                return {
                    "is_abnormal": True,
                    "detection_result": "异常",
                    "abnormal_events": abnormal_events,
                    "reason": resp1.get("原因分析", "") + f"\n事件分类：{resp2}",
                    "confidence": "高",
                    "detection_level": 2
                }
            
            # ========== 第三级：偶发事件深度判断 ==========
            logger.info("🔍 检测到偶发事件，进入第三级深度判断...")
            response = llm_inference(query, self.system_prompt3)
            logger.info(f"第三级检测结果:\n{response}")
            
            resp3 = parse_json_response(response)
            if not resp3:
                logger.error("❌ 第三级检测结果解析失败，保守判定为异常")
                return {
                    "is_abnormal": True,
                    "detection_result": "异常",
                    "abnormal_events": abnormal_events,
                    "reason": resp1.get("原因分析", ""),
                    "confidence": "低",
                    "detection_level": 2
                }
            
            # 第三级的最终判断
            if resp3.get("判断结果") == "正常":
                logger.info("✅ 第三级判定：正常（偶发但合理）")
                # 更新规则库（可选）
                try:
                    self.update_rules(resp3)
                except Exception as e:
                    logger.warning(f"⚠️ 规则更新失败：{e}")
                
                return {
                    "is_abnormal": False,
                    "detection_result": "正常",
                    "abnormal_events": [],
                    "reason": resp3.get("判断依据", "偶发事件，但符合校园常规"),
                    "confidence": "中",
                    "detection_level": 3,
                    "new_rule": resp3.get("通用规则", "")
                }
            else:
                logger.info("🚨 第三级判定：异常")
                return {
                    "is_abnormal": True,
                    "detection_result": "异常",
                    "abnormal_events": abnormal_events,
                    "reason": resp3.get("判断依据", resp1.get("原因分析", "")),
                    "confidence": "高",
                    "detection_level": 3
                }
        
        except Exception as e:
            logger.error(f"❌ 异常检测失败：{e}", exc_info=True)
            return None
    
    def update_rules(self, resp: Dict):
        """
        更新规则库（当检测到新的正常行为模式时）
        
        Args:
            resp: 包含新规则的响应字典
        """
        try:
            new_rule = resp.get("通用规则", "")
            if not new_rule:
                logger.warning("⚠️ 新规则为空，跳过更新")
                return
            
            # 读取现有规则
            old_rules = load_txt(rules_file)
            
            # 合并规则
            merged_rules = llm_inference(
                old_rules + "\n新增规则：" + new_rule,
                self.rule_merging_prompt
            )
            
            # 保存更新后的规则
            save_txt(rules_file, merged_rules)
            logger.info(f"✅ 规则库已更新：{new_rule}")
            
        except Exception as e:
            logger.error(f"❌ 更新规则失败：{e}")


# 全局单例
_detector_instance = None


def get_detector():
    """获取异常检测器的全局单例"""
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = AnomalyDetector(
            system_prompt1=llm_prompt1,
            system_prompt2=llm_prompt2,
            system_prompt3=llm_prompt3,
            rule_update_prompt=rule_merging_prompt
        )
    
    return _detector_instance


if __name__ == "__main__":
    # 测试代码
    detector = get_detector()
    
    # 测试正常场景
    test_desc_normal = """
    摄像头camera1在20250623_094842的监控视频分析报告：
    一个穿着蓝色T恤的男生在教学楼走廊内正常行走，
    手持书包，步伐正常，没有异常行为。
    """
    
    result = detector.detect(test_desc_normal)
    print("\n测试正常场景：")
    print(result)
    
    # 测试异常场景
    test_desc_abnormal = """
    摄像头camera2在20250623_094850的监控视频分析报告：
    两名男性在走廊内发生激烈争执，其中一人推搡另一人，
    并有拳头挥舞的动作，周围学生纷纷避让。
    """
    
    result = detector.detect(test_desc_abnormal)
    print("\n测试异常场景：")
    print(result)

