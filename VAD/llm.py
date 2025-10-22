from utils import *
from prompts import llm_prompt1, llm_prompt2, llm_prompt3, rule_merging_prompt
from rag import HybridRetriever

class SurveillanceLLMAnalyzer:
    def __init__(self, system_prompt1, system_prompt2, system_prompt3, rule_update_prompt):
        self.system_prompt1 = system_prompt1
        self.system_prompt2 = system_prompt2
        self.system_prompt3 = system_prompt3
        self.rule_merging_prompt = rule_update_prompt
        # self.retriever = HybridRetriever(w_faiss=0.6, w_bm25=0.4)
        # self.retriever.build(json_load_file=train_desc_file, index_file=faiss_file, bm25_keyword_file=bm25_file)
        # self.retriever.load(index_file=faiss_file, bm25_keyword_file=bm25_file)
        
    def update_rules(self, resp):
        try:
            old_rules = load_txt(rules_file)
            latest_rules = llm_inference(old_rules + "\n新增规则:" + resp["通用规则"], self.rule_merging_prompt)
            save_txt(rules_file, latest_rules)
        except Exception as e:
            print(f"更新规则失败: {e}")

    def analyze(self, surveillance_log):
        query = f"以下是当前你需要进行异常检测的监控视频描述：\n{surveillance_log}"
        if not surveillance_log:
            print("❌ 错误：监控描述不能为空！")
            return None

        try:
            # 第一步：异常检测
            # history = self.retriever.retrieve(surveillance_log, top_k=5)
            # history = "\n".join([f"{i + 1}. {event}" for i, event in enumerate(history)])
            # history_text = f"""以下是一些可供参考的历史异常检测结果为正常的一些例子：\n{history}\n"""
            response = llm_inference(query, self.system_prompt1)
            print("第一次推理结果:\n", response)
            resp1 = parse_json_response(response)

            if resp1.get("异常检测结果") == "异常":
                # 第二步：频繁 or 偶发判断
                events_text = "\n".join([f"{i + 1}. {event}" for i, event in enumerate(resp1.get("异常事件", []))])
                response = llm_inference(f"以下为多个行为事件，请分类：\n{events_text}", self.system_prompt2)
                print("第二次推理结果:\n", response)
                resp2 = parse_json_response(response)

                # 第三步：异常检测
                if "偶发事件" in response:
                    response = llm_inference(query, self.system_prompt3)
                    print("第三次推理结果:\n", response)
                    resp3 = parse_json_response(response)

                    if resp3.get("判断结果") == "正常":
                        self.update_rules(resp3)
                    return resp3
                else:
                    return resp2
            else:
                return resp1

        except Exception as e:
            print(f"❌ 检测失败: {e}")
            return None

    def batch_analyze(self, json_load_file, json_save_file):
        logs = load_json(json_load_file)
        results = {}

        for key, value in logs.items():
            print(f"\n📝 正在处理 {key} ...")
            result = self.analyze(value)
            if result:
                results[key] = result
                print(f"✅ {key} 处理完成:\n{result}")

        save_json(json_save_file, results)
        print(f"\n🚀 批量处理完成，结果保存到 {json_save_file}")

if __name__ == "__main__":
    analyzer = SurveillanceLLMAnalyzer(
        system_prompt1=llm_prompt1,
        system_prompt2=llm_prompt2,
        system_prompt3=llm_prompt3,
        rule_update_prompt=rule_merging_prompt
    )

    analyzer.batch_analyze(test_desc_file, test_results_file)
