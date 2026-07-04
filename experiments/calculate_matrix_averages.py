#!/usr/bin/env python3
"""
计算矩阵平均值并添加到报告
"""

import numpy as np

# 简单一致性率矩阵
agreement_matrix = np.array([
    [1.000, 0.942, 0.888, 0.932, 0.913, 0.934, 0.753],
    [0.942, 1.000, 0.879, 0.948, 0.906, 0.934, 0.747],
    [0.888, 0.879, 1.000, 0.893, 0.887, 0.884, 0.727],
    [0.932, 0.948, 0.893, 1.000, 0.920, 0.932, 0.741],
    [0.913, 0.906, 0.887, 0.920, 1.000, 0.928, 0.756],
    [0.934, 0.934, 0.884, 0.932, 0.928, 1.000, 0.766],
    [0.753, 0.747, 0.727, 0.741, 0.756, 0.766, 1.000]
])

# Cohen's Kappa 矩阵
kappa_matrix = np.array([
    [1.000, 0.865, 0.729, 0.836, 0.786, 0.845, 0.405],
    [0.865, 1.000, 0.710, 0.877, 0.769, 0.846, 0.394],
    [0.729, 0.710, 1.000, 0.732, 0.708, 0.718, 0.315],
    [0.836, 0.877, 0.732, 1.000, 0.796, 0.835, 0.353],
    [0.786, 0.769, 0.708, 0.796, 1.000, 0.820, 0.373],
    [0.845, 0.846, 0.718, 0.835, 0.820, 1.000, 0.431],
    [0.405, 0.394, 0.315, 0.353, 0.373, 0.431, 1.000]
])

judger_names = [
    "harmbench_judger",
    "gpt_judger_contextual_harmbench", 
    "gpt_judger_harmful_binary",
    "gpt_judger_harmbench_style",
    "gpt_judger_openai_policy",
    "gpt_judger_tap_style",
    "rejection_prefix_judger"
]

def calculate_row_averages():
    """计算每行的平均值（排除对角线元素）"""
    
    print("## 简单一致性率矩阵")
    print("| 评判器 | harmbench_judger | gpt_judger_contextual_harmbench | gpt_judger_harmful_binary | gpt_judger_harmbench_style | gpt_judger_openai_policy | gpt_judger_tap_style | rejection_prefix_judger | **平均** |")
    print("|--------|------------------|----------------------------------|---------------------------|----------------------------|--------------------------|----------------------|-------------------------|----------|")
    
    for i, judger in enumerate(judger_names):
        row = agreement_matrix[i]
        # 计算平均值（排除对角线元素）
        avg = np.mean([row[j] for j in range(7) if j != i])
        
        row_str = f"| {judger[:20]} |"
        for val in row:
            row_str += f" {val:.3f} |"
        row_str += f" **{avg:.3f}** |"
        print(row_str)
    
    print()
    print("## Cohen's Kappa 矩阵")
    print("| 评判器 | harmbench_judger | gpt_judger_contextual_harmbench | gpt_judger_harmful_binary | gpt_judger_harmbench_style | gpt_judger_openai_policy | gpt_judger_tap_style | rejection_prefix_judger | **平均** |")
    print("|--------|------------------|----------------------------------|---------------------------|----------------------------|--------------------------|----------------------|-------------------------|----------|")
    
    for i, judger in enumerate(judger_names):
        row = kappa_matrix[i]
        # 计算平均值（排除对角线元素）
        avg = np.mean([row[j] for j in range(7) if j != i])
        
        row_str = f"| {judger[:20]} |"
        for val in row:
            row_str += f" {val:.3f} |"
        row_str += f" **{avg:.3f}** |"
        print(row_str)

if __name__ == "__main__":
    calculate_row_averages()