"""
24点游戏A*求解算法
使用A*搜索算法求解24点问题，并生成详细的求解过程用于训练数据
"""

import heapq
import copy
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from fractions import Fraction


@dataclass
class GameState:
    """游戏状态，包含当前剩余的数字列表"""
    numbers: List[Fraction]
    path: List[str]  # 记录到达此状态的操作路径
    cost: int  # 已使用的步数
    
    def __hash__(self):
        return hash(tuple(sorted(self.numbers)))
    
    def __eq__(self, other):
        return sorted(self.numbers) == sorted(other.numbers)
    
    def __lt__(self, other):
        return self.cost < other.cost


class Game24Solver:
    """24点游戏求解器"""
    
    def __init__(self, target: int = 24, epsilon: float = 1e-9):
        self.target = target
        self.epsilon = epsilon
        self.operations = ['+', '-', '*', '/']
        
    def is_target(self, numbers: List[Fraction]) -> bool:
        """检查是否达到目标状态"""
        if len(numbers) != 1:
            return False
        return abs(float(numbers[0]) - self.target) < self.epsilon
    
    def heuristic(self, state: GameState) -> float:
        """启发式函数：估计到达目标的代价"""
        if len(state.numbers) == 1:
            return abs(float(state.numbers[0]) - self.target)
        
        # 计算当前数字与目标的最小距离
        min_dist = float('inf')
        for num in state.numbers:
            dist = abs(float(num) - self.target)
            min_dist = min(min_dist, dist)
        
        # 考虑剩余步数
        remaining_steps = len(state.numbers) - 1
        return min_dist + remaining_steps
    
    def get_next_states(self, state: GameState) -> List[GameState]:
        """获取所有可能的下一步状态"""
        next_states = []
        numbers = state.numbers
        
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                a, b = numbers[i], numbers[j]
                remaining = [numbers[k] for k in range(len(numbers)) if k != i and k != j]
                
                # 尝试所有运算
                operations_results = [
                    ('+', a + b, f"({a} + {b})"),
                    ('-', a - b, f"({a} - {b})"),
                    ('-', b - a, f"({b} - {a})"),
                    ('*', a * b, f"({a} * {b})"),
                ]
                
                # 除法需要检查分母不为0
                if abs(float(b)) > self.epsilon:
                    operations_results.append(('/', a / b, f"({a} / {b})"))
                if abs(float(a)) > self.epsilon:
                    operations_results.append(('/', b / a, f"({b} / {a})"))
                
                for op, result, operation_str in operations_results:
                    new_numbers = remaining + [result]
                    new_path = state.path + [f"计算 {operation_str} = {result}"]
                    new_state = GameState(
                        numbers=new_numbers,
                        path=new_path,
                        cost=state.cost + 1
                    )
                    next_states.append(new_state)
        
        return next_states
    
    def solve_with_path(self, initial_numbers: List[float]) -> Optional[Tuple[bool, List[str]]]:
        """
        使用A*算法求解24点问题，返回是否有解和详细求解路径
        """
        # 转换为分数以避免浮点数精度问题
        fractions = [Fraction(num).limit_denominator() for num in initial_numbers]
        
        initial_state = GameState(
            numbers=fractions,
            path=[f"初始数字: {initial_numbers}"],
            cost=0
        )
        
        # A*搜索
        open_set = [(self.heuristic(initial_state), initial_state)]
        closed_set: Set[GameState] = set()
        explored_paths = []  # 记录所有探索过的路径，用于生成long CoT
        
        while open_set:
            _, current_state = heapq.heappop(open_set)
            
            if current_state in closed_set:
                continue
            
            closed_set.add(current_state)
            explored_paths.append(f"探索状态: {[float(n) for n in current_state.numbers]}")
            
            # 检查是否达到目标
            if self.is_target(current_state.numbers):
                success_path = current_state.path + [f"成功！得到结果: {float(current_state.numbers[0])}"]
                return True, success_path + [f"总共探索了 {len(explored_paths)} 个状态"]
            
            # 生成下一步状态
            next_states = self.get_next_states(current_state)
            
            for next_state in next_states:
                if next_state not in closed_set:
                    # 添加回溯信息（模拟人类思考过程中的反思）
                    if len(next_state.numbers) == 1 and not self.is_target(next_state.numbers):
                        next_state.path.append(f"这个结果 {float(next_state.numbers[0])} 不是24，需要回溯")
                    
                    priority = next_state.cost + self.heuristic(next_state)
                    heapq.heappush(open_set, (priority, next_state))
        
        # 无解的情况，返回探索过程
        failure_path = [f"初始数字: {initial_numbers}"] + explored_paths + ["经过详细搜索，无法找到解法"]
        return False, failure_path


def generate_24_problems(count: int = 100) -> List[List[int]]:
    """生成24点问题"""
    import random
    problems = []
    
    # 生成一些已知有解的问题
    known_solvable = [
        [3, 3, 8, 8],  # (8/(3-8/3)) = 24
        [4, 1, 8, 7],  # (8-4) * (7-1) = 24
        [1, 1, 8, 8],  # (8-1/1)*8 = 24
        [2, 2, 10, 10], # (10+10+2+2) = 24
        [1, 2, 3, 4],  # (1+2+3)*4 = 24
        [1, 5, 5, 5],  # 5*(5-1/5) = 24
        [2, 3, 4, 6],  # 2*3*(6-4) = 24
        [1, 1, 2, 12], # (1+1)*12 = 24
        [3, 4, 6, 6],  # 6*6-4-3 = 24
        [2, 4, 4, 6],  # (6+4-2)*4 = 24
    ]
    
    problems.extend(known_solvable)
    
    # 随机生成更多问题
    for _ in range(count - len(known_solvable)):
        problem = [random.randint(1, 13) for _ in range(4)]
        problems.append(problem)
    
    return problems[:count]


if __name__ == "__main__":
    # 测试求解器
    solver = Game24Solver()
    
    # 测试已知有解的问题
    test_problem = [3, 3, 8, 8]
    print(f"测试问题: {test_problem}")
    
    has_solution, path = solver.solve_with_path(test_problem)
    print(f"是否有解: {has_solution}")
    print("求解路径:")
    for step in path:
        print(f"  {step}")