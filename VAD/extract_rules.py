from utils import *
from prompts import rule_extraction_prompt, rule_merging_prompt

class RuleGenerator:
    def __init__(self, extraction_prompt, merging_prompt):
        self.extraction_prompt = extraction_prompt
        self.merging_prompt = merging_prompt

    def generate(self, json_load_file, txt_save_file):
        descriptions = load_json(json_load_file)
        rule_extraction = []

        for key, value in descriptions.items():
            print(f"📝 正在提取 {key} 的规则...")
            rule = llm_inference(value, self.extraction_prompt)
            rule_extraction.append(rule)
            print(f"{key} 提取完成:\n{rule}")

        save_txt(txt_save_file, "\n".join(rule_extraction))
        print(f"\n✅ 规则提取完成，保存到 {txt_save_file}")

    def merge(self, txt_load_file, txt_save_file):
        origin_rules = load_txt(txt_load_file)
        print(f"🔄 正在合并规则...")

        merged_rules = llm_inference(origin_rules, self.merging_prompt)
        print(f"\n合并后的规则:\n{merged_rules}")

        save_txt(txt_save_file, merged_rules)
        print(f"\n✅ 规则合并完成，保存到 {txt_save_file}")

if __name__ == "__main__":
    generator = RuleGenerator(
        extraction_prompt=rule_extraction_prompt,
        merging_prompt=rule_merging_prompt
    )

    # generator.generate(train_desc_file, rules_origin_file)

    # 合并所有规则
    generator.merge(rules_origin_file, rules_file)
