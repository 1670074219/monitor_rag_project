import time
import gradio as gr
from llm import SurveillanceLLMAnalyzer
from vlm import SurveillanceVLMAnalyzer
from prompts import *

class SurveillanceApp:
    def __init__(self):
        self.output = ""
        self.output2 = ""
        self.video = ""
        self.llm_analyzer = SurveillanceLLMAnalyzer(
            system_prompt1=llm_prompt1,
            system_prompt2=llm_prompt2,
            system_prompt3=llm_prompt3,
            rule_update_prompt=rule_merging_prompt
        )
        self.vlm_analyzer = SurveillanceVLMAnalyzer(system_prompt=vlm_prompt)
        self.rules = rules  # 从 prompts 导入

    def upload_video(self, video_path, up_video):
        self.video = video_path if video_path else up_video
        return '开始检测', "⬆️Upload & Start Chat"

    def gradio_ask(self, text_input, chatbot):
        result = self.vlm_analyzer.analyze(self.video)
        self.output = f"监控视频描述:{result}"

        result = self.llm_analyzer.analyze(self.output)
        answer = ""
        for key, value in result.items():
            answer += f"{key}:{value}\n"
        self.output2 = answer

        return text_input, chatbot

    def gradio_answer(self, chatbot):
        chatbot.append(("监控视频描述", self.output))
        chatbot.append(("异常检测结果", self.output2))
        return '开始检测', chatbot

    def gradio_reset(self):
        return "", None, None, None, "⬆️Upload & Start Chat"

    def gradio_ask2(self, text_input, chatbot):
        result = self.llm_analyzer.analyze(text_input)
        answer = ""
        for key, value in result.items():
            answer += f"{key}:{value}\n"
        self.output = answer
        
        return text_input, chatbot

    def gradio_answer2(self, text_input, chatbot):
        chatbot.append((text_input, self.output))
        return None, chatbot

    def gradio_reset2(self):
        return "", None

    def get_current_time(self):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    def update_time(self):
        return gr.update(value=self.get_current_time())

    def update_rules(self):
        return gr.Textbox(
            value=f"{self.rules}",
            label="Normal Rules",
            lines=13,
            interactive=False,
            container=True
        )

    def run(self):
        with gr.Blocks(title="Anomaly Detection",
                       css="#time_display{background: rgba(0, 0, 0, 1); color:yellow;font-size: 20px} button{color:white} .gradio-container {background: url('file=neu3.jpg'); background-size: 100% 100%;width:500}") as demo:
            gr.HTML("""
                <div style="display: flex; align-items: center; justify-content: center;">
                    <h1 style="color: write;">面向开放世界的场景监控视频异常检测系统</h1>
                    <img src="file/neu.jpg" alt="logo" width="60" height="60" style="margin-left: 10px;">
                </div>
            """)
            gr.HTML("""
                <div style="margin-top: 10px; margin-bottom: 10px;">
                    <h3 style="color: white;">📝 操作说明：</h3>
                    <ul style="color: white; line-height: 1.6;">
                        <li>1. 请上传视频或目录，然后点击按钮 <b>⬆️Upload & Start Chat</b></li>
                        <li>2. 发送消息请点击按钮 <b>💭Send</b></li>
                        <li>3. 清空历史请点击按钮 <b>🗑️Clear</b></li>
                    </ul>
                </div>
            """)
            time_display = gr.Button(interactive=False, elem_id="time_display")
            demo.load(self.update_time, inputs=None, outputs=time_display, every=1)

            with gr.Tabs():
                with gr.Tab("🎬 视频处理"):
                    with gr.Row():
                        with gr.Column(scale=1, visible=True):
                            with gr.Tab("Video", elem_id='video_tab'):
                                up_video = gr.Video(interactive=True, include_audio=False, elem_id="video_upload", height=290)
                                video_path = gr.Textbox(show_label=False, placeholder='📁 输入或粘贴视频路径', interactive=True, container=False)
                            upload_button = gr.Button(value="⬆️Upload & Start Chat", interactive=True, elem_id="upload_button", variant="primary")

                        with gr.Column(scale=1, visible=True):
                            chatbot = gr.Chatbot(elem_id="chatbot", label='💡 异常检测对话')
                            with gr.Row():
                                with gr.Column(scale=14):
                                    text_input = gr.Textbox(show_label=False, placeholder='💬 请先上传视频，然后输入消息...', interactive=True, container=False)
                                with gr.Column(scale=3, min_width=0):
                                    run = gr.Button("💭Send")
                                with gr.Column(scale=3, min_width=0):
                                    clear = gr.Button("🗑️Clear")

                    upload_button.click(self.upload_video, [up_video, video_path], [text_input, upload_button])
                    text_input.submit(self.gradio_ask, [text_input, chatbot], [text_input, chatbot]).then(
                        self.gradio_answer, [chatbot], [text_input, chatbot]
                    )
                    run.click(self.gradio_ask, [text_input, chatbot], [text_input, chatbot]).then(
                        self.gradio_answer, [chatbot], [text_input, chatbot]
                    )
                    run.click(lambda: "", None, text_input)
                    clear.click(self.gradio_reset, [], [chatbot, up_video, video_path, text_input, upload_button], queue=False)

                with gr.Tab("📜 文本处理"):
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                show_rules = gr.Textbox(
                                    value=f"{self.rules}",
                                    label="📚 正常行为规则",
                                    lines=13,
                                    interactive=False,
                                    container=True
                                )
                                rules_update = gr.Button("🔄Update", variant="primary")
                            chatbot2 = gr.Chatbot(elem_id="chatbot", label='🧠 异常检测对话')

                        with gr.Row():
                            with gr.Column(scale=14):
                                text_input2 = gr.Textbox(show_label=False, placeholder='💬 请输入文本...', interactive=True, container=False)
                            with gr.Column(scale=3, min_width=0):
                                run2 = gr.Button("💭Send")
                            with gr.Column(scale=3, min_width=0):
                                clear2 = gr.Button("🗑️Clear")

                    rules_update.click(self.update_rules, inputs=[], outputs=[show_rules])
                    text_input2.submit(self.gradio_ask2, [text_input2, chatbot2], [text_input2, chatbot2]).then(
                        self.gradio_answer2, [text_input2, chatbot2], [text_input2, chatbot2]
                    )
                    run2.click(self.gradio_ask2, [text_input2, chatbot2], [text_input2, chatbot2]).then(
                        self.gradio_answer2, [text_input2, chatbot2], [text_input2, chatbot2]
                    )
                    run2.click(lambda: "", None, '')
                    clear2.click(self.gradio_reset2, [], [chatbot2, text_input2], queue=False)

        demo.launch(allowed_paths=["."], share=True)

if __name__ == "__main__":
    app = SurveillanceApp()
    app.run()
