"""注释：3D 打印实验的并行 LLM 辅助 BO 工具与采样函数集合。"""

import numpy as np  # 数值计算与数组类型
from openai import OpenAI  # OpenAI 客户端，用于调用 LLM
import json  # 解析与生成 JSON 字符串

def _sample_one_candidate_AM(args):
    """注释：根据历史观测和目标分数，采样下一组 3D 打印参数。"""
    history_variant_str, target_score = args  # 解包历史记录与目标分数
    prompt = f"""  # 构造提示词（Prompt）
    The following are past evaluations of the stringing percentage and their corresponding Nozzle Temperature and Z Hop values:    
    {history_variant_str}

    You are allowed to adjust **only five slicing parameters**:
    1. **Nozzle Temperature**: Range 220–260°C (step: 1°C)
    2. **Z Hop Height**: Range 0.1–1.0 mm (step: 0.1 mm)
    3. **Coasting Volume**:	0.02–0.1 mm³ (step: 0.01 mm³)
    4. **Retraction Distance**: 1.0–10.0 mm (step: 1 mm)
    5. **Outer Wall Wipe Distance**: 0.0–1.0 mm (step: 0.1 mm)
    
    These slicing settings are fixed:
    - Retraction Speed = 60 mm/s
    - Travel Speed = 178 mm/s
    - Fan Speed = 60 %
    
    Other slicing settings are set to be the software's default values.
        
    Recommend a new ([Nozzle Temperature (°C), Z Hop Height (mm), Coasting Volume (mm³), Retraction Distance (mm), Outer Wall Wipe Distance (mm)) that can achieve the stringing percentage of {target_score}.
        
    **Instructions:**
    - Return only one 5D vector: `[Nozzle Temperature (°C), Z Hop Height (mm), Coasting Volume (mm³), Retraction Distance (mm), Outer Wall Wipe Distance (mm)]`
    - Ensure the values respect the allowed ranges and increments.
    - Respond with strictly valid **JSON format**.
    - Do **not** include any explanations, comments, or extra text. Do not include the word jason.
    """

    client = OpenAI()  # 初始化 OpenAI 客户端

    while True:  # 循环直到成功解析到合法结果
        try:  # 捕获解析/类型错误以重试
            messages = [  # 构造对话消息
                {
                    "role": "system",
                    "content": "You are an AI assistant that helps me optimizing the 3D manufacturing process by controlling parameters.",
                },
                {"role": "user", "content": prompt},
            ]
            response = client.chat.completions.create(  # 发起 LLM 调用
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=50,
            ).choices[0].message.content.strip()  # 取出回复文本
            print(response)  # 打印原始回复便于调试
            extracted_value = json.loads(response)  # 解析 JSON 列表
            if isinstance(extracted_value, list) and len(extracted_value) == 5:  # 校验维度
                extracted_value = [np.float64(v) for v in extracted_value]  # 转换为 float64
                return tuple(extracted_value)  # 返回参数元组

        except (ValueError, json.JSONDecodeError):  # JSON 或类型错误则重试
            continue  # 进入下一次循环


def _predict_llm_score_AM(args):
    """注释：基于历史观测与候选参数，预测字符串缺陷比例。"""
    x, history_variant_str = args  # 解包候选参数与历史记录
    prompt = f"""  # 构造预测提示词
    The following are past evaluations of the stringing percentage and the corresponding Nozzle Temperature and Z hop.    
    {history_variant_str}
    You are allowed to adjust **only five slicing parameters**:
    1. **Nozzle Temperature**: Range 220–260°C (step: 1°C)
    2. **Z Hop Height**: Range 0.1–1.0 mm (step: 0.1 mm)
    3. **Coasting Volume**:	0.02–0.1 mm³ (step: 0.01 mm³)
    4. **Retraction Distance**: 1.0–10.0 mm (step: 1 mm)
    5. **Outer Wall Wipe Distance**: 0.0–1.0 mm (step: 0.1 mm)
    
    All other slicing settings are fixed:
    - Retraction Speed = 60 mm/s
    - Travel Speed = 178 mm/s
    - Fan Speed = 60 %
    
    Predict the stringing percentage at ([Nozzle Temperature, Z Hop Height, Coasting Volume, Retraction Distance, Outer Wall Wipe Distance) = {x}.
    
    The stringing percentage needs to be a single value between 0 to 100. 
    Return only a single numerical value. Do not include any explanations, labels, formatting, percentage symbol, or extra text. 
    The response must be strictly a valid floating-point number.
    """

    client = OpenAI()  # 初始化 OpenAI 客户端
    while True:  # 循环直到得到可解析的数值
        try:  # 捕获无法转为浮点数的情况
            messages = [  # 构造对话消息
                {
                    "role": "system",
                    "content": "You are an AI assistant that helps me optimizing the 3D manufacturing process by controlling parameters.",
                },
                {"role": "user", "content": prompt},
            ]
            response = client.chat.completions.create(  # 发起 LLM 调用
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=50,
            ).choices[0].message.content.strip()  # 取出回复文本
            print(response)  # 打印原始回复便于调试
            return float(response), tuple(x)  # 返回预测值与参数
        except ValueError:  # 转换失败则重试
            continue  # 进入下一次循环
