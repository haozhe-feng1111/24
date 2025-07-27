"""
生成24点游戏的long CoT训练数据
"""

import json
import random
from typing import List, Dict
from game24_solver import Game24Solver, generate_24_problems
import os


class LongCoTDataGenerator:
    """Long Chain of Thought数据生成器"""
    
    def __init__(self):
        self.solver = Game24Solver()
    
    def problem_to_cot_format(self, numbers: List[int], solution_path: List[str], has_solution: bool) -> Dict:
        """将问题和求解路径转换为CoT格式"""
        
        # 构建问题描述
        question = f"请解决24点游戏：给定数字{numbers}，使用加、减、乘、除运算（每个数字只能使用一次，可使用括号），使得最终结果为24。"
        
        # 构建思考过程（long CoT）
        thinking_process = "让我仔细分析这个24点问题：\n\n"
        
        # 添加初始分析
        thinking_process += f"给定数字：{numbers}\n"
        thinking_process += "目标：通过四则运算得到24\n"
        thinking_process += "策略：使用A*搜索算法，系统性地探索所有可能的运算组合\n\n"
        
        # 添加详细的求解步骤
        thinking_process += "详细求解过程：\n"
        for i, step in enumerate(solution_path, 1):
            thinking_process += f"步骤{i}: {step}\n"
        
        # 添加反思和验证
        if has_solution:
            thinking_process += "\n验证结果：\n"
            thinking_process += "让我验证最终答案是否正确...\n"
            thinking_process += "经过验证，计算结果确实等于24。\n"
            answer = "找到解决方案！通过上述运算可以得到24。"
        else:
            thinking_process += "\n详细分析：\n"
            thinking_process += "经过系统性的搜索，我已经尝试了所有可能的运算组合。\n"
            thinking_process += "在考虑了加减乘除的所有排列组合后，没有找到能够得到24的解法。\n"
            answer = "经过详细分析，这组数字无法通过四则运算得到24。"
        
        return {
            "question": question,
            "thinking": thinking_process,
            "answer": answer,
            "has_solution": has_solution,
            "numbers": numbers
        }
    
    def enhance_cot_with_exploration(self, cot_data: Dict) -> Dict:
        """增强CoT数据，添加更多探索过程和人类思维模式"""
        numbers = cot_data["numbers"]
        
        # 添加更多的思考元素
        enhanced_thinking = f"""让我来解决这个24点问题：数字 {numbers}

首先，我需要分析这些数字的特点：
- 数字范围：最小值{min(numbers)}，最大值{max(numbers)}
- 数字总和：{sum(numbers)}
- 是否有重复数字：{'是' if len(set(numbers)) < len(numbers) else '否'}

我的解题策略：
1. 尝试大数乘小数的组合
2. 寻找能构成24的因数分解
3. 考虑使用括号改变运算优先级
4. 如果直接运算困难，尝试构造中间结果

现在开始详细搜索：

{cot_data['thinking']}

总结我的思考过程：
- 我系统性地考虑了各种运算组合
- 使用了启发式搜索来提高效率
- 在每一步都验证了计算结果
- 当遇到死路时及时回溯
"""
        
        enhanced_data = cot_data.copy()
        enhanced_data["thinking"] = enhanced_thinking
        return enhanced_data
    
    def generate_dataset(self, num_problems: int = 1000) -> List[Dict]:
        """生成训练数据集"""
        print(f"开始生成{num_problems}个训练样本...")
        
        dataset = []
        problems = generate_24_problems(num_problems)
        
        for i, problem in enumerate(problems):
            if i % 100 == 0:
                print(f"已生成 {i}/{num_problems} 个样本")
            
            # 求解问题
            has_solution, solution_path = self.solver.solve_with_path(problem)
            
            # 转换为CoT格式
            cot_data = self.problem_to_cot_format(problem, solution_path, has_solution)
            
            # 增强CoT数据
            enhanced_data = self.enhance_cot_with_exploration(cot_data)
            
            dataset.append(enhanced_data)
        
        print(f"数据生成完成！总共{len(dataset)}个样本")
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str):
        """保存数据集到文件"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"数据集已保存到: {filename}")
        
        # 统计信息
        total_samples = len(dataset)
        solvable_samples = sum(1 for d in dataset if d['has_solution'])
        unsolvable_samples = total_samples - solvable_samples
        
        print(f"统计信息:")
        print(f"  总样本数: {total_samples}")
        print(f"  有解样本: {solvable_samples} ({solvable_samples/total_samples*100:.1f}%)")
        print(f"  无解样本: {unsolvable_samples} ({unsolvable_samples/total_samples*100:.1f}%)")


def main():
    """主函数"""
    generator = LongCoTDataGenerator()
    
    # 生成训练数据
    print("=== 生成训练数据 ===")
    train_dataset = generator.generate_dataset(num_problems=1000)
    generator.save_dataset(train_dataset, "data/train/train_data.json")
    
    # 生成测试数据
    print("\n=== 生成测试数据 ===")
    test_dataset = generator.generate_dataset(num_problems=200)
    generator.save_dataset(test_dataset, "data/test/test_data.json")
    
    # 生成一些样本用于检查
    print("\n=== 样本检查 ===")
    sample = train_dataset[0]
    print("问题:", sample["question"])
    print("\n思考过程 (前500字符):")
    print(sample["thinking"][:500] + "..." if len(sample["thinking"]) > 500 else sample["thinking"])
    print("\n答案:", sample["answer"])


if __name__ == "__main__":
    main()