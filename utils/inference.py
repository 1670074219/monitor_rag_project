import openai
from PIL import Image
import numpy as np
import cv2
import io

openai_client = openai.OpenAI(api_key="token-abc123", base_url="http://localhost:8000/v1")

response = openai_client.chat.completions.create(
    model="qwen2.5",
    max_tokens=2048,
    messages=[
        {"role": "system", "content": "请判断以下视频描述是否属于异常行为（如吵架、斗殴、手持危险物品、奇怪行为如昏倒、打架等）。如果描述涉及暴力、威胁、危险或不寻常的行为，请标记为异常行为。如果没有涉及这些内容，则标记为正常行为。回答是不是异常行为，并给出原因。"},
        {"role": "user", "content": "以下是当前你需要进行异常检测的监控视频描述：\n{人数：2\n\n每个人的外貌特征：\n- 人物1：穿着深色短袖上衣和黑色裤子，头发较短，体型适中。\n- 人物2：穿着浅色短袖上衣和深色裤子，头发较长，体型适中。\n\n每个人的行为动作：\n- 人物1：从画面左侧进入走廊，向右走动。\n- 人物2：从画面右侧进入走廊，向左走动，然后转身离开镜头。\n\n与环境的交互：无交互\n\n视频总结：摄像头camera1在20250623_095918的监控视频分析报告：一个穿着深色短袖上衣和黑色裤子的人在走廊中向右走动并转身离开镜头，一个穿着浅色短袖上衣和深色裤子的人在走廊中向左走动并转身离开镜头。他们与环境没有发生任何交互。}"}
    ]
)
print(response.choices[0].message.content)