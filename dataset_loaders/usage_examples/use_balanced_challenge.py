#!/usr/bin/env python3
"""
平衡挑战数据集使用示例
====================

演示如何使用Phase1实验结果驱动的平衡挑战数据集。
包含50个防御挑战样本(高ASR)和50个攻击挑战样本(低ASR)。
"""

import json
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataset_loaders import DatasetFactory
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def main():
    """主函数：演示平衡挑战数据集的使用"""
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]平衡挑战数据集使用示例[/bold blue]\n"
        "基于Phase1实验结果，包含攻防双重挑战样本",
        style="cyan"
    ))
    
    try:
        # 1. 加载配置文件
        config_path = "dataset_loaders/usage_examples/configs/balanced_challenge.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        console.print(f"\n📁 加载配置文件: {config_path}")
        
        # 2. 创建数据集加载器
        console.print("\n🔧 创建平衡挑战数据集加载器...")
        loader = DatasetFactory.create_loader(config, console)
        
        # 3. 加载提示数据
        console.print("\n📊 加载平衡挑战数据集...")
        try:
            prompts = loader.load_prompts()
            console.print(f"✅ 成功加载 {len(prompts)} 个提示")
        except Exception as e:
            console.print(f"⚠️  数据集加载失败: {e}")
            console.print("💡 这通常意味着需要先运行Phase1分析器生成分析结果")
            console.print("   运行命令: python experiments/core/phase1_analyzer.py")
            return
        
        # 4. 获取数据集信息
        info = loader.get_dataset_info()
        
        # 5. 显示数据集基本信息
        console.print(f"\n📋 数据集基本信息:")
        basic_table = Table(show_header=True, header_style="bold magenta")
        basic_table.add_column("属性")
        basic_table.add_column("值")
        
        basic_table.add_row("数据集名称", info['name'])
        basic_table.add_row("描述", info['description'])
        basic_table.add_row("版本", info['version'])
        basic_table.add_row("总样本数", str(info['total_samples']))
        
        console.print(basic_table)
        
        # 6. 显示样本组成
        composition = info['composition']
        comp_table = Table(show_header=True, header_style="bold green")
        comp_table.add_column("挑战类型")
        comp_table.add_column("样本数量")
        comp_table.add_column("占比")
        comp_table.add_column("用途说明")
        
        comp_table.add_row(
            "防御挑战", 
            str(composition['defense_challenge_samples']),
            f"{composition['defense_challenge_ratio']:.1%}",
            "高ASR - 测试防御能力"
        )
        comp_table.add_row(
            "攻击挑战", 
            str(composition['attack_challenge_samples']),
            f"{composition['attack_challenge_ratio']:.1%}",
            "低ASR - 测试攻击能力"
        )
        
        console.print(f"\n🎯 样本组成分析:")
        console.print(comp_table)
        
        # 7. 显示ASR统计
        asr_stats = info['asr_statistics']
        asr_table = Table(show_header=True, header_style="bold yellow")
        asr_table.add_column("ASR统计")
        asr_table.add_column("高ASR样本")
        asr_table.add_column("低ASR样本")
        
        asr_table.add_row(
            "平均ASR", 
            f"{asr_stats['high_asr_mean']:.3f}",
            f"{asr_stats['low_asr_mean']:.3f}"
        )
        asr_table.add_row(
            "ASR范围",
            f"{asr_stats['high_asr_range'][0]:.3f} - {asr_stats['high_asr_range'][1]:.3f}",
            f"{asr_stats['low_asr_range'][0]:.3f} - {asr_stats['low_asr_range'][1]:.3f}"
        )
        
        console.print(f"\n📈 ASR分布统计:")
        console.print(asr_table)
        
        # 8. 显示数据集分布
        dataset_dist = info['dataset_distribution']
        if dataset_dist:
            dist_table = Table(show_header=True, header_style="bold cyan")
            dist_table.add_column("源数据集")
            dist_table.add_column("防御挑战样本")
            dist_table.add_column("攻击挑战样本")
            dist_table.add_column("总计")
            
            for dataset, counts in dataset_dist.items():
                total = counts['defense_challenge'] + counts['attack_challenge']
                dist_table.add_row(
                    dataset,
                    str(counts['defense_challenge']),
                    str(counts['attack_challenge']),
                    str(total)
                )
            
            console.print(f"\n🗂️  源数据集分布:")
            console.print(dist_table)
        
        # 9. 显示样本示例
        console.print(f"\n📝 样本示例 (前5个):")
        for i in range(min(5, len(prompts))):
            metadata = loader.get_sample_metadata(i)
            challenge_type = metadata.get('challenge_type', 'unknown')
            asr = metadata.get('avg_asr', 0.0)
            
            style = "red" if challenge_type == 'high_asr' else "green"
            console.print(f"[{style}]{i+1}. [{challenge_type}] ASR: {asr:.3f}[/{style}]")
            console.print(f"   {prompts[i][:100]}...")
            console.print()
        
        # 10. 使用建议
        console.print(Panel.fit(
            "[bold green]使用建议[/bold green]\n\n"
            "• [bold]防御挑战样本[/bold]: 用于测试防御机制的有效性\n"
            "• [bold]攻击挑战样本[/bold]: 用于测试攻击方法的突破能力\n"
            "• [bold]平衡设计[/bold]: 50:50的比例确保攻防测试的平衡性\n"
            "• [bold]数据质量[/bold]: 基于Phase1实验结果的科学选择\n\n"
            "[italic]在Phase4实验中替换原有的100样本数据集，提高实验效率[/italic]",
            style="green"
        ))
        
        # 11. 导出选项演示
        console.print(f"\n💾 数据集导出选项:")
        export_options = [
            ("JSON格式", "完整的数据集信息和样本"),
            ("CSV格式", "便于数据分析的表格形式"),
            ("JSONL格式", "流式处理友好的格式")
        ]
        
        for format_name, description in export_options:
            console.print(f"• [bold]{format_name}[/bold]: {description}")
        
        console.print(f"\n💡 导出示例:")
        console.print(f"   loader.export_dataset('output/balanced_challenge.json', 'json')")
        console.print(f"   loader.export_dataset('output/balanced_challenge.csv', 'csv')")
        
        console.print(f"\n✅ [bold green]平衡挑战数据集演示完成![/bold green]")
        
    except Exception as e:
        console.print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()