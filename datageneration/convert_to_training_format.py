"""
将生成的数据转换为训练框架所需的格式
"""

import json
import os
from typing import List, Dict


def convert_to_sharegpt_format(data: List[Dict]) -> List[Dict]:
    """将数据转换为ShareGPT格式"""
    converted_data = []
    
    for item in data:
        # 构建对话格式
        conversation = {
            "messages": [
                {
                    "role": "user",
                    "content": item["question"]
                },
                {
                    "role": "assistant", 
                    "content": f"{item['thinking']}\n\n{item['answer']}"
                }
            ]
        }
        converted_data.append(conversation)
    
    return converted_data


def main():
    """转换训练和测试数据"""
    
    # 转换训练数据
    print("转换训练数据...")
    with open("../../data/train/train_data.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    train_converted = convert_to_sharegpt_format(train_data)
    
    with open("../../data/train/train_data.json", "w", encoding="utf-8") as f:
        json.dump(train_converted, f, ensure_ascii=False, indent=2)
    
    print(f"训练数据转换完成，共{len(train_converted)}条")
    
    # 转换测试数据
    print("转换测试数据...")
    with open("../../data/test/test_data.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    test_converted = convert_to_sharegpt_format(test_data)
    
    with open("../../data/test/test_data.json", "w", encoding="utf-8") as f:
        json.dump(test_converted, f, ensure_ascii=False, indent=2)
    
    print(f"测试数据转换完成，共{len(test_converted)}条")
    
    # 检查转换结果
    print("\n样本检查:")
    sample = train_converted[0]
    print("用户输入:", sample["messages"][0]["content"][:100] + "...")
    print("\n助手回复 (前200字符):")
    print(sample["messages"][1]["content"][:200] + "...")


if __name__ == "__main__":
    main()