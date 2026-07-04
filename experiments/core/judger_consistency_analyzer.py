#!/usr/bin/env python3
"""
Phase1 Judger Consistency Analysis
计算7个评判器的两两一致性率，识别高一致性组别并推荐代表性评判器

Usage:
    python judger_consistency_analyzer.py
"""

import json
import os
import numpy as np
from itertools import combinations
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import cohen_kappa_score

class JudgerConsistencyAnalyzer:
    """分析Phase1实验结果中7个评判器的一致性"""
    
    def __init__(self, phase1_dir: str):
        self.phase1_dir = phase1_dir
        self.judger_names = [
            "harmbench_judger",
            "gpt_judger_contextual_harmbench", 
            "gpt_judger_harmful_binary",
            "gpt_judger_harmbench_style",
            "gpt_judger_openai_policy",
            "gpt_judger_tap_style",
            "rejection_prefix_judger"
        ]
        self.all_results = []
        self.agreement_matrix = np.zeros((7, 7))
        self.kappa_matrix = np.zeros((7, 7))
        self.agreement_counts = np.zeros((7, 7))
        
    def extract_all_results(self) -> None:
        """从所有87个实验文件中提取评判器结果"""
        print("🔍 提取Phase1实验结果...")
        
        files = [f for f in os.listdir(self.phase1_dir) if f.endswith('.json')]
        print(f"找到 {len(files)} 个实验文件")
        
        total_samples = 0
        for file_name in files:
            file_path = os.path.join(self.phase1_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if data.get('status') != 'success':
                    print(f"⚠️  跳过失败的实验: {file_name}")
                    continue
                
                # 提取文件元信息
                model = data['config']['model']
                dataset = data['config']['dataset']
                
                # 提取每个样本的评判器结果
                for sample in data.get('sample_results', []):
                    if sample.get('status') == 'success':
                        judger_results = sample.get('judger_result_on_clean')
                        if judger_results and len(judger_results) == 7:
                            result_record = {
                                'model': model,
                                'dataset': dataset,
                                'sample_index': sample.get('sample_index'),
                                'file_name': file_name
                            }
                            # 按固定顺序提取评判器分数
                            for judger in self.judger_names:
                                result_record[judger] = judger_results.get(judger)
                            
                            self.all_results.append(result_record)
                            total_samples += 1
                            
            except Exception as e:
                print(f"❌ 处理文件 {file_name} 时出错: {e}")
        
        print(f"✅ 成功提取 {total_samples} 个样本的评判器结果")
        print(f"📊 覆盖 {len(set(r['model'] for r in self.all_results))} 个模型")
        print(f"📊 覆盖 {len(set(r['dataset'] for r in self.all_results))} 个数据集")
    
    def calculate_pairwise_agreement(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """计算所有评判器对的一致性率和Cohen's Kappa"""
        print("\n🧮 计算评判器间的两两一致性...")
        
        # 构建评判器结果矩阵
        judger_scores = np.array([[r[judger] for judger in self.judger_names] 
                                 for r in self.all_results])
        
        print(f"分析 {len(judger_scores)} 个样本的评判结果")
        
        # 计算所有评判器对的一致性率和Cohen's Kappa
        agreement_matrix = np.zeros((7, 7))
        kappa_matrix = np.zeros((7, 7))
        
        for i, judger1 in enumerate(self.judger_names):
            for j, judger2 in enumerate(self.judger_names):
                if i == j:
                    agreement_matrix[i][j] = 1.0  # 自己与自己100%一致
                    kappa_matrix[i][j] = 1.0  # Kappa也是1.0
                else:
                    # 计算简单一致性率
                    agreements = (judger_scores[:, i] == judger_scores[:, j]).sum()
                    agreement_rate = agreements / len(judger_scores)
                    agreement_matrix[i][j] = agreement_rate
                    
                    # 计算Cohen's Kappa
                    try:
                        kappa = cohen_kappa_score(judger_scores[:, i], judger_scores[:, j])
                        kappa_matrix[i][j] = kappa
                    except:
                        # 如果计算失败（比如所有预测都相同），设为0
                        kappa_matrix[i][j] = 0.0
        
        # 转换为DataFrame便于展示
        agreement_df = pd.DataFrame(
            agreement_matrix, 
            index=self.judger_names, 
            columns=self.judger_names
        )
        
        kappa_df = pd.DataFrame(
            kappa_matrix,
            index=self.judger_names,
            columns=self.judger_names
        )
        
        self.agreement_matrix = agreement_matrix
        self.kappa_matrix = kappa_matrix
        return agreement_df, kappa_df
    
    def identify_consistent_groups(self, threshold: float = 0.8, kappa_threshold: float = 0.6) -> Dict[str, Any]:
        """识别高一致性评判器组 (基于简单一致性率和Cohen's Kappa)"""
        print(f"\n🎯 识别一致性评判器组...")
        print(f"简单一致性率阈值: >{threshold*100}%")
        print(f"Cohen's Kappa阈值: >{kappa_threshold}")
        
        high_consistency_pairs = []
        high_kappa_pairs = []
        
        # 找出所有高一致性的评判器对
        for i in range(7):
            for j in range(i+1, 7):
                agreement_rate = self.agreement_matrix[i][j]
                kappa_score = self.kappa_matrix[i][j]
                
                if agreement_rate >= threshold:
                    high_consistency_pairs.append({
                        'judger1': self.judger_names[i],
                        'judger2': self.judger_names[j], 
                        'agreement_rate': agreement_rate,
                        'kappa_score': kappa_score
                    })
                
                if kappa_score >= kappa_threshold:
                    high_kappa_pairs.append({
                        'judger1': self.judger_names[i],
                        'judger2': self.judger_names[j], 
                        'agreement_rate': agreement_rate,
                        'kappa_score': kappa_score
                    })
        
        print(f"\n📊 简单一致性率 >{threshold*100}% 的评判器对 ({len(high_consistency_pairs)} 个):")
        for pair in high_consistency_pairs:
            print(f"  • {pair['judger1'][:20]:<20} ↔ {pair['judger2'][:20]:<20} : {pair['agreement_rate']:.3f} (κ={pair['kappa_score']:.3f})")
        
        print(f"\n📊 Cohen's Kappa >{kappa_threshold} 的评判器对 ({len(high_kappa_pairs)} 个):")
        for pair in high_kappa_pairs:
            print(f"  • {pair['judger1'][:20]:<20} ↔ {pair['judger2'][:20]:<20} : κ={pair['kappa_score']:.3f} (简单一致性={pair['agreement_rate']:.3f})")
        
        # 基于连通性分析构建一致性组
        groups = self._build_consistency_groups(high_consistency_pairs, threshold)
        
        return {
            'threshold': threshold,
            'kappa_threshold': kappa_threshold,
            'high_consistency_pairs': high_consistency_pairs,
            'high_kappa_pairs': high_kappa_pairs,
            'groups': groups
        }
    
    def _build_consistency_groups(self, pairs: List[Dict], threshold: float) -> List[List[str]]:
        """基于高一致性对构建评判器组"""
        # 构建邻接列表
        adjacency = defaultdict(set)
        for pair in pairs:
            adjacency[pair['judger1']].add(pair['judger2'])
            adjacency[pair['judger2']].add(pair['judger1'])
        
        # 使用深度优先搜索找连通分量
        visited = set()
        groups = []
        
        for judger in self.judger_names:
            if judger not in visited:
                group = []
                self._dfs(judger, adjacency, visited, group)
                groups.append(group)
        
        return groups
    
    def _dfs(self, judger: str, adjacency: Dict, visited: set, group: List[str]):
        """深度优先搜索找连通分量"""
        visited.add(judger)
        group.append(judger)
        
        for neighbor in adjacency[judger]:
            if neighbor not in visited:
                self._dfs(neighbor, adjacency, visited, group)
    
    def recommend_representatives(self, groups: List[List[str]]) -> List[Dict[str, Any]]:
        """为每个一致性组推荐代表性评判器"""
        print(f"\n🎖️  为 {len(groups)} 个组别推荐代表性评判器...")
        
        recommendations = []
        
        for i, group in enumerate(groups):
            if len(group) == 1:
                # 单独的评判器，直接推荐
                recommendations.append({
                    'group_id': i + 1,
                    'group_members': group,
                    'representative': group[0],
                    'reason': '独立评判器，无高一致性伙伴',
                    'avg_agreement_with_others': self._calculate_avg_agreement_with_others(group[0])
                })
            else:
                # 多个评判器的组，选择最具代表性的
                best_representative = self._select_best_representative(group)
                recommendations.append({
                    'group_id': i + 1,
                    'group_members': group,
                    'representative': best_representative['judger'],
                    'reason': best_representative['reason'],
                    'avg_agreement_with_others': best_representative['avg_agreement']
                })
        
        return recommendations
    
    def _calculate_avg_agreement_with_others(self, judger: str) -> float:
        """计算某个评判器与其他所有评判器的平均一致性"""
        judger_idx = self.judger_names.index(judger)
        agreements = []
        
        for i in range(7):
            if i != judger_idx:
                agreements.append(self.agreement_matrix[judger_idx][i])
        
        return np.mean(agreements)
    
    def _select_best_representative(self, group: List[str]) -> Dict[str, Any]:
        """为一致性组选择最佳代表评判器"""
        best_judger = None
        best_score = -1
        best_reason = ""
        
        for judger in group:
            # 计算与组外评判器的平均一致性 (作为多样性指标)
            avg_agreement_outside = self._calculate_avg_agreement_with_others(judger)
            
            # 简单启发式: 选择与组外评判器平均一致性最高的 (更有代表性)
            if avg_agreement_outside > best_score:
                best_score = avg_agreement_outside
                best_judger = judger
                best_reason = f"与组外评判器平均一致性最高 ({avg_agreement_outside:.3f})"
        
        return {
            'judger': best_judger,
            'reason': best_reason,
            'avg_agreement': best_score
        }
    
    def generate_report(self) -> str:
        """生成完整的分析报告"""
        print("\n📋 生成分析报告...")
        
        # 执行分析
        agreement_df, kappa_df = self.calculate_pairwise_agreement()
        consistency_analysis = self.identify_consistent_groups()
        recommendations = self.recommend_representatives(consistency_analysis['groups'])
        
        # 生成报告
        report = []
        report.append("# Phase1 评判器一致性分析报告")
        report.append("=" * 50)
        
        # 基本统计
        report.append(f"\n## 数据概览")
        report.append(f"- 实验文件数: 87个")
        report.append(f"- 样本总数: {len(self.all_results)}")
        report.append(f"- 评判器数量: 7个")
        report.append(f"- 模型数: {len(set(r['model'] for r in self.all_results))}")
        report.append(f"- 数据集数: {len(set(r['dataset'] for r in self.all_results))}")
        
        # 评判器列表
        report.append(f"\n## 评判器列表")
        for i, judger in enumerate(self.judger_names, 1):
            report.append(f"{i}. {judger}")
        
        # 一致性指标说明
        report.append(f"\n## 一致性指标说明")
        report.append("**简单一致性率**: 两个评判器给出相同结果的样本比例 (0.0-1.0)")
        report.append("**Cohen's Kappa**: 校正偶然一致性后的一致性系数")
        report.append("  - κ > 0.8: 几乎完全一致")
        report.append("  - κ 0.6-0.8: 高度一致")
        report.append("  - κ 0.4-0.6: 中等一致")
        report.append("  - κ 0.2-0.4: 较低一致")
        report.append("  - κ < 0.2: 轻微一致")
        
        # 简单一致性矩阵
        report.append(f"\n## 简单一致性率矩阵")
        report.append("```")
        report.append(agreement_df.round(3).to_string())
        report.append("```")
        
        # Cohen's Kappa矩阵
        report.append(f"\n## Cohen's Kappa 矩阵")
        report.append("```")
        report.append(kappa_df.round(3).to_string())
        report.append("```")
        
        # 高一致性对 - 简单一致性率
        report.append(f"\n## 高一致性评判器对分析")
        report.append(f"\n### 简单一致性率 >80% ({len(consistency_analysis['high_consistency_pairs'])} 对)")
        if consistency_analysis['high_consistency_pairs']:
            for pair in consistency_analysis['high_consistency_pairs']:
                report.append(f"- {pair['judger1']} ↔ {pair['judger2']}: {pair['agreement_rate']:.3f} (κ={pair['kappa_score']:.3f})")
        else:
            report.append("❌ 未发现一致性率超过80%的评判器对")
        
        # 高一致性对 - Cohen's Kappa
        report.append(f"\n### Cohen's Kappa >0.6 ({len(consistency_analysis['high_kappa_pairs'])} 对)")
        if consistency_analysis['high_kappa_pairs']:
            for pair in consistency_analysis['high_kappa_pairs']:
                report.append(f"- {pair['judger1']} ↔ {pair['judger2']}: κ={pair['kappa_score']:.3f} (简单一致性={pair['agreement_rate']:.3f})")
        else:
            report.append("❌ 未发现Cohen's Kappa超过0.6的评判器对")
        
        # 一致性组别
        report.append(f"\n## 一致性组别分析")
        for i, group in enumerate(consistency_analysis['groups'], 1):
            report.append(f"\n### 组别 {i} ({len(group)} 个评判器)")
            for judger in group:
                report.append(f"  - {judger}")
        
        # 代表性评判器推荐
        report.append(f"\n## 代表性评判器推荐")
        for rec in recommendations:
            report.append(f"\n### 组别 {rec['group_id']} 代表: {rec['representative']}")
            report.append(f"- 组员: {', '.join(rec['group_members'])}")
            report.append(f"- 推荐理由: {rec['reason']}")
            report.append(f"- 与其他评判器平均一致性: {rec['avg_agreement_with_others']:.3f}")
        
        # 最终建议
        report.append(f"\n## 最终建议")
        selected_judgers = [rec['representative'] for rec in recommendations]
        report.append(f"建议选择以下 {len(selected_judgers)} 个代表性评判器用于后续实验:")
        for i, judger in enumerate(selected_judgers, 1):
            report.append(f"{i}. {judger}")
        
        # 指标对比分析
        report.append(f"\n## 指标对比分析")
        
        # 计算平均一致性
        avg_simple_agreement = np.mean(self.agreement_matrix[np.triu_indices(7, k=1)])
        avg_kappa = np.mean(self.kappa_matrix[np.triu_indices(7, k=1)])
        
        report.append(f"- 平均简单一致性率: {avg_simple_agreement:.3f}")
        report.append(f"- 平均Cohen's Kappa: {avg_kappa:.3f}")
        
        # 找出两种指标差异最大的评判器对
        diff_matrix = self.agreement_matrix - self.kappa_matrix
        max_diff_indices = np.unravel_index(np.argmax(diff_matrix), diff_matrix.shape)
        max_diff = diff_matrix[max_diff_indices]
        
        report.append(f"- 最大指标差异: {self.judger_names[max_diff_indices[0]]} ↔ {self.judger_names[max_diff_indices[1]]}")
        report.append(f"  简单一致性: {self.agreement_matrix[max_diff_indices]:.3f}, Kappa: {self.kappa_matrix[max_diff_indices]:.3f} (差异: {max_diff:.3f})")
        
        # 节省的评判器数量
        original_count = 7
        recommended_count = len(selected_judgers) 
        saved_count = original_count - recommended_count
        if saved_count > 0:
            report.append(f"\n💰 通过选择代表性评判器，可以减少 {saved_count} 个评判器 ({saved_count/original_count*100:.1f}%)")
        
        # 补充说明
        report.append(f"\n## 补充说明")
        report.append("1. **简单一致性率**高但**Cohen's Kappa**低的情况，通常表明存在较多偶然一致")
        report.append("2. **Cohen's Kappa**提供了更保守和准确的一致性评估")
        report.append("3. 推荐基于两种指标综合考虑进行最终评判器选择")
        report.append("4. 在重要实验中，建议使用多个代表性评判器进行交叉验证")
        
        return "\n".join(report)


def main():
    """主函数"""
    phase1_dir = os.environ.get(
        "PROMPTSECURITY_PHASE1_DIR",
        "experiments/placeholders_phase1",
    )
    
    if not os.path.exists(phase1_dir):
        print(f"❌ Phase1实验目录不存在: {phase1_dir}")
        return
    
    # 创建分析器并执行分析
    analyzer = JudgerConsistencyAnalyzer(phase1_dir)
    
    try:
        # 1. 提取所有结果
        analyzer.extract_all_results()
        
        if not analyzer.all_results:
            print("❌ 未找到有效的实验结果")
            return
        
        # 2. 生成分析报告
        report = analyzer.generate_report()
        
        # 3. 保存报告
        report_path = os.path.join(os.path.dirname(phase1_dir), "judger_consistency_analysis_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 4. 显示结果
        print("\n" + "="*60)
        print(report)
        print("\n" + "="*60)
        print(f"📄 完整报告已保存至: {report_path}")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
