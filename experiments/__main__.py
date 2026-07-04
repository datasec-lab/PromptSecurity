#!/usr/bin/env python3
"""
PromptSecurity 实验系统 - 占位符统一执行模式

所有实验默认使用占位符模式执行：
- python -m experiments --model gpt-4o --attack ArtPrompt  # 自动创建+执行
- python -m experiments --model gpt-4o --attack ArtPrompt --generate-only  # 仅创建
- python -m experiments --run-placeholders  # 执行现有占位符
"""

import sys
import json
import logging
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.core.unified_interface import PromptSecurityInterface, create_cli_parser


def setup_logging(verbose=False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # 设置根日志记录器
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 禁用HTTP请求日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    
    # 禁用HTTP和网络请求日志
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("requests.packages.urllib3").setLevel(logging.WARNING)
    
    # 禁用HuggingFace相关日志
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)
    logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.WARNING)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)
    logging.getLogger("transformers.models").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub.utils").setLevel(logging.WARNING)
    
    # 禁用bitsandbytes日志（量化库）
    logging.getLogger("bitsandbytes").setLevel(logging.WARNING)


def is_success_result(result: dict) -> bool:
    status = result.get("status")
    if status == "success":
        return True
    if status != "completed":
        return False
    failed_samples = result.get("failed_samples")
    total_samples = result.get("total_samples")
    if failed_samples is not None:
        return failed_samples == 0 and (total_samples or 0) > 0
    sample_results = result.get("sample_results")
    if isinstance(sample_results, list) and sample_results:
        return all(s.get("status") == "success" for s in sample_results)
    return False
    logging.getLogger("bitsandbytes.cextension").setLevel(logging.WARNING)
    
    # 禁用transformers的进度条和模型加载日志
    import os
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # 禁用tqdm进度条显示
    try:
        import tqdm
        tqdm.tqdm.disable = True
    except ImportError:
        pass
    
    # 减少其他噪音日志
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("torch.distributed").setLevel(logging.WARNING)
    logging.getLogger("torch.nn").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)
    logging.getLogger("tqdm").setLevel(logging.WARNING)
    
    # 禁用matplotlib相关日志（包括font_manager的DEBUG消息）
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)
    
    # 禁用模型加载器的详细日志，但允许本地模型的调试日志
    logging.getLogger("models.model_loader").setLevel(logging.WARNING)
    # 启用本地模型的调试日志以诊断无响应问题
    logging.getLogger("models.local_models.local_huggingface_model").setLevel(logging.DEBUG)
    
    # 设置环境变量来禁用进度条
    import os
    if not verbose:
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def print_banner():
    """打印欢迎信息"""
    print("""
================================================================
                    PromptSecurity 实验系统
                    
   统一实验接口 | 模块自动发现 | 灵活配置管理 | 分阶段实验
================================================================
    """)


def list_available_methods(interface, component_type=None):
    """列出可用方法"""
    
    methods = interface.list_available_methods(component_type)
    
    if component_type:
        print(f"\n{component_type.upper()} 可用方法:")
        if isinstance(methods, dict):
            for category, method_list in methods.items():
                if isinstance(method_list, list):
                    print(f"  {category}: {len(method_list)} 个")
                    for method in method_list:
                        print(f"    - {method}")
                else:
                    print(f"  {category}: {method_list}")
        elif isinstance(methods, list):
            print(f"  共 {len(methods)} 个:")
            for method in methods:
                print(f"    - {method}")
    else:
        print("\n所有可用方法:")
        total_count = 0
        
        for comp_type, comp_methods in methods.items():
            if isinstance(comp_methods, dict):
                # 计算总数
                count = 0
                if "all" in comp_methods:
                    count = len(comp_methods["all"])
                else:
                    count = sum(len(v) if isinstance(v, list) else 1 
                               for v in comp_methods.values())
                
                print(f"  {comp_type.upper()}: {count} 个")
                total_count += count
                
                # 显示分类
                for category, method_list in comp_methods.items():
                    if isinstance(method_list, list) and category != "all":
                        print(f"    {category}: {len(method_list)} 个")
            
            elif isinstance(comp_methods, list):
                print(f"  {comp_type.upper()}: {len(comp_methods)} 个")
                total_count += len(comp_methods)
        
        print(f"\n总计: {total_count} 个可用方法")


def generate_template_config(template_name, output_path):
    """生成模板配置文件"""
    
    config_manager = get_config_manager()
    templates = config_manager.generate_template_configs()
    
    if template_name not in templates:
        print(f"错误: 未知模板 '{template_name}'")
        print(f"可用模板: {list(templates.keys())}")
        return False
    
    try:
        config_manager.save_config(templates[template_name], output_path)
        print(f"✅ 模板配置已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"❌ 保存模板失败: {e}")
        return False


def show_method_info(interface, method_name, method_type):
    """显示方法详细信息"""
    
    info = interface.get_method_info(method_name, method_type)
    
    if not info:
        print(f"❌ 未找到方法信息: {method_type}/{method_name}")
        return
    
    print(f"\n方法信息: {method_name}")
    print(f"类型: {method_type}")
    print(f"配置文件: {info['config_path']}")
    print(f"配置内容:")
    print(json.dumps(info['config'], indent=2, ensure_ascii=False))
    
    # 显示兼容信息
    compatible = interface.get_compatible_methods(method_name, method_type)
    if compatible:
        print(f"\n兼容的组件:")
        for comp_type, comp_list in compatible.items():
            print(f"  {comp_type}: {len(comp_list)} 个")


def show_examples():
    """显示使用示例"""
    
    examples = """
使用示例:

1. 基础评估:
   python -m experiments --model gpt-4o --attack ArtPrompt --defense no_defense

2. 使用配置文件:
   python -m experiments --config my_experiment.json

3. Phase 1 实验:
   python -m experiments --phase 1 --sample-limit 30

4. 指定随机种子：
   python -m experiments --model gpt-4o --attack ArtPrompt --seed 123

5. 批量实验:
   python -m experiments --experiment-type batch --config batch_config.json

6. 列出可用方法:
   python -m experiments --list attacks
   python -m experiments --list all

7. 生成模板配置:
   python -m experiments --generate-template basic_evaluation config.json

8. 查看方法信息:
   python -m experiments --info ArtPrompt attack

9. 并行执行:
   python -m experiments --config config.json --parallel --workers 4

10. 环境变量配置:
   export PS_MODEL=gpt-4o
   export PS_ATTACK=ArtPrompt
   python -m experiments

11. 使用多个judger:
    python -m experiments --model gpt-4o --attack no_attack --defense no_defense --judger harmbench_judger gpt_judger_harmful_binary

12. 批量实验（多个模型）:
    python -m experiments --model gpt-4o claude-3-5-sonnet --attack no_attack ArtPrompt --defense no_defense --sample-limit 10

13. 完整批量实验:
    python -m experiments --model gpt-4o claude-3-5-sonnet --attack no_attack ArtPrompt GPTFUZZER --defense no_defense smooth_llm --dataset harmbench jbb --sample-limit 5

14. 运行占位符实验（详细输出）:
    python -m experiments --run-placeholders --verbose
    python -m experiments --run-placeholders --verbose --max-length 300 --run-model gpt-4o
    python -m experiments --run-placeholders --verbose --run-attack ArtPrompt --run-limit 5
    
    # 重跑失败的实验
    python -m experiments --run-placeholders --placeholder-status failed
    python -m experiments --run-placeholders --placeholder-status failed --verbose
    
    # 只运行特定状态的实验
    python -m experiments --run-placeholders --placeholder-status pending
    python -m experiments --run-placeholders --placeholder-status created

14. 查看结果看板:
    python -m experiments --dashboard
    python -m experiments --dashboard --dashboard-filter gpt-4o
    python -m experiments --dashboard --dashboard-model gpt-4o --dashboard-limit 10
    python -m experiments --dashboard --dashboard-dataset harmbench --dashboard-sort asr
    python -m experiments --dashboard --dashboard-attack no_attack --dashboard-sort clean_asr
    python -m experiments --dashboard --dashboard-sort model --dashboard-limit 10
    python -m experiments --dashboard --dashboard-sort dataset
    python -m experiments --dashboard --dashboard-sort attack
    python -m experiments --dashboard --dashboard-sort defense
    
15. 从其他目录读取placeholders:
    python -m experiments --dashboard --placeholders-dir experiments/placeholders_phase1
    python -m experiments --dashboard --placeholders-dir /path/to/custom/placeholders

配置文件格式 (JSON):
{
  "model": "gpt-4o",
  "attack": "ArtPrompt", 
  "defense": "smooth_llm",
  "dataset": "harmbench",
  "judger": "harmbench_judger",
  "sample_limit": 100,
  "seed": 42,
  "experiment_name": "我的实验"
}

配置文件格式 (YAML):
model: gpt-4o
attack: ArtPrompt
defense: smooth_llm
dataset: harmbench
judger: harmbench_judger
sample_limit: 100
seed: 42
experiment_name: 我的实验
    """
    
    print(examples)


def main():
    """主函数"""
    
    try:
        # 创建解析器
        parser = create_cli_parser()
        
        # 添加额外的功能选项
        parser.add_argument("--info", nargs=2, metavar=("METHOD_NAME", "METHOD_TYPE"),
                          help="显示方法详细信息")
        parser.add_argument("--show-examples", action="store_true", help="显示使用示例")
        parser.add_argument("--banner", action="store_true", help="显示欢迎信息")
        parser.add_argument("--dashboard", action="store_true", help="显示实验结果看板")
        parser.add_argument("--placeholders-dir", metavar="DIR", help="指定读取placeholders的目录 (默认: experiments/placeholders)")
        parser.add_argument("--dashboard-filter", metavar="PATTERN", help="过滤看板结果文件名")
        parser.add_argument("--dashboard-model", metavar="MODEL", help="过滤特定模型")
        parser.add_argument("--dashboard-dataset", metavar="DATASET", help="过滤特定数据集")
        parser.add_argument("--dashboard-attack", metavar="ATTACK", help="过滤特定攻击")
        parser.add_argument("--dashboard-defense", metavar="DEFENSE", help="过滤特定防御")
        parser.add_argument("--dashboard-limit", type=int, metavar="N", help="限制显示条数")
        parser.add_argument("--dashboard-sort", choices=["asr", "clean_asr", "samples", "time", "model", "dataset", "attack", "defense", "created_time", "status", "success_rate"], 
                          default="time", help="排序方式 (默认: time)")
        parser.add_argument("--dashboard-only-asr", action="store_true",
                          help="只显示ASR有结果(非N/A)的占位符实验")
        
        # 占位符相关参数
        parser.add_argument("--placeholder-mode", action="store_true", help="强制使用占位符模式")
        
        # 快捷预设
        parser.add_argument("--baseline", action="store_true", help="基线安全评估 (no_attack + no_defense)")
        parser.add_argument("--security-test", action="store_true", help="安全攻击测试 (ArtPrompt + no_defense)")
        parser.add_argument("--defense-eval", action="store_true", help="防御效果评估 (GPTFUZZER + smooth_llm)")
        parser.add_argument("--resume", action="store_true", help="恢复中断的实验 (自动运行failed状态的占位符)")
        
        parser.add_argument("--list-placeholders", action="store_true", help="列出所有占位符实验")
        parser.add_argument("--placeholder-status", choices=["pending", "running", "in_progress", "success", "failed", "created", "completed"], 
                          help="过滤占位符状态")
        parser.add_argument("--run-placeholders", action="store_true", help="运行占位符实验")
        parser.add_argument("--run-model", metavar="MODEL", help="只运行指定模型的占位符实验")
        parser.add_argument("--run-attack", metavar="ATTACK", help="只运行指定攻击的占位符实验")
        parser.add_argument("--run-defense", metavar="DEFENSE", help="只运行指定防御的占位符实验")
        parser.add_argument("--run-dataset", metavar="DATASET", help="只运行指定数据集的占位符实验")
        parser.add_argument("--run-judger", metavar="JUDGER", help="只运行指定评判器的占位符实验")
        parser.add_argument("--run-limit", type=int, metavar="N", help="限制运行的占位符实验数量")
        parser.add_argument("--generate-only", action="store_true", help="仅生成占位符，不执行实验")
        parser.add_argument("--max-length", type=int, default=200, help="详细输出模式下文本显示的最大长度 (默认: 200)")
        parser.add_argument("--rerun-failed", action="store_true",
                          help="重跑除success外的所有样本状态（如 completed/failed/pending）")
        parser.add_argument("--force-rerun", action="store_true",
                          help="强制重跑全部样本（包含success和completed）")

        args = parser.parse_args()

        # show_examples()

        # 显示欢迎信息
        if args.banner or (not any(vars(args).values())):
            print_banner()

        # 设置日志
        setup_logging(args.verbose)

        # 显示使用示例
        if args.show_examples:
            return
        
        # 显示结果看板
        if args.dashboard:
            from experiments.core.placeholder_dashboard import UnifiedDashboard, PlaceholderDashboard
            
            # 确定placeholder目录
            placeholders_dir = args.placeholders_dir or "experiments/placeholders"
            print(f"📁 读取placeholder目录: {placeholders_dir}")
            
            unified_dashboard = UnifiedDashboard(placeholders_dir=placeholders_dir)
            print(unified_dashboard.display_unified_view(sort_by=args.dashboard_sort, only_asr=args.dashboard_only_asr))
            
            if any([args.dashboard_model, args.dashboard_attack, args.dashboard_defense, 
                   args.dashboard_dataset, args.dashboard_filter]):
                print("\n" + "="*80 + "\n🔍 过滤结果:")
                placeholder_dashboard = PlaceholderDashboard(placeholders_dir=placeholders_dir)
                placeholder_dashboard.load_placeholders()
                filtered_placeholders = placeholder_dashboard.filter_placeholders(
                    model=args.dashboard_model, attack=args.dashboard_attack,
                    defense=args.dashboard_defense, dataset=args.dashboard_dataset
                )
                if args.dashboard_filter:
                    filtered_placeholders = [p for p in filtered_placeholders 
                                           if args.dashboard_filter in p.get('_filename', '')]
                placeholder_dashboard.placeholders = filtered_placeholders
                print(placeholder_dashboard.display_table(
                    sort_by=args.dashboard_sort, limit=args.dashboard_limit, show_details=True, only_asr=args.dashboard_only_asr))
            return
        
        # 列出可用方法
        if args.list:
            interface = PromptSecurityInterface()
            list_available_methods(interface, None if args.list == "all" else args.list)
            return
        
        # 显示方法信息
        if args.info:
            interface = PromptSecurityInterface()
            method_name, method_type = args.info
            show_method_info(interface, method_name, method_type)
            return
        
        # 占位符相关功能
        if args.list_placeholders:
            from experiments.core.placeholder_system import ExperimentPlaceholder
            placeholder_manager = ExperimentPlaceholder(seed=getattr(args, 'seed', 42))
            placeholders = placeholder_manager.list_placeholders(args.placeholder_status)
            
            print(f"\n占位符实验列表 ({len(placeholders)} 个):")
            print("=" * 100)
            
            for p in placeholders:
                config = p["config"]
                status_icon = {"pending": "⏳", "running": "🔄", "success": "✅", "failed": "❌"}.get(p["status"], "❓")
                
                print(f"{status_icon} {p['filename']}")
                print(f"   配置: {config.get('model', 'N/A')} + {config.get('attack', 'N/A')} + {config.get('defense', 'N/A')} + {config.get('dataset', 'N/A')} + {config.get('judger', 'N/A')}")
                print(f"   状态: {p['status']} | 样本: {p['total_samples']} | 成功: {p['success_count']} | 失败: {p['failed_count']}")
                print(f"   ID: {p['experiment_id']}")
                print("-" * 100)
            return
        
        if args.run_placeholders:
            from experiments.core.placeholder_system import ExperimentPlaceholder
            from experiments.core.placeholder_runner import PlaceholderExperimentRunner
            
            placeholder_manager = ExperimentPlaceholder(seed=getattr(args, 'seed', 42))
            runner = PlaceholderExperimentRunner(
                verbose=args.verbose,
                max_length=args.max_length,
                seed=getattr(args, 'seed', 42),
                rerun_failed=args.rerun_failed,
                force_rerun=args.force_rerun
            )
            
            # 获取可执行状态的占位符
            if args.placeholder_status:
                # 如果指定了状态过滤，只获取该状态的占位符
                placeholders = placeholder_manager.list_placeholders(args.placeholder_status)
                if not placeholders:
                    print(f"✅ 没有 {args.placeholder_status} 状态的占位符实验")
                    return
                if args.verbose:
                    print(f"📋 找到 {len(placeholders)} 个 {args.placeholder_status} 状态的占位符")
            else:
                # 默认状态选择会受到重跑策略影响
                all_placeholders = placeholder_manager.list_placeholders()
                if args.force_rerun:
                    placeholders = all_placeholders
                    if args.verbose:
                        print(f"📋 force-rerun 启用：选中全部状态，占位符 {len(placeholders)} 个")
                elif args.rerun_failed:
                    placeholders = [p for p in all_placeholders if p.get("status") != "success"]
                    if args.verbose:
                        print(f"📋 rerun-failed 启用：选中非success状态，占位符 {len(placeholders)} 个")
                else:
                    runnable_statuses = {"running", "in_progress", "created", "pending", "failed"}
                    placeholders = [p for p in all_placeholders if p.get("status") in runnable_statuses]
                    if args.verbose:
                        print(f"📋 默认模式：选中待执行状态，占位符 {len(placeholders)} 个")

                if not placeholders:
                    if args.force_rerun:
                        print("✅ 没有可重跑的占位符实验")
                    elif args.rerun_failed:
                        print("✅ 没有非success状态的占位符实验")
                    else:
                        print("✅ 没有待执行、运行中或失败的占位符实验")
                    return
            
            # 应用过滤条件
            filtered_placeholders = []
            for p in placeholders:
                config = p.get("config", {})
                
                # 检查各种过滤条件
                if args.run_model and config.get("model") != args.run_model:
                    continue
                if args.run_attack and config.get("attack") != args.run_attack:
                    continue
                if args.run_defense and config.get("defense") != args.run_defense:
                    continue
                if args.run_dataset and config.get("dataset") != args.run_dataset:
                    continue
                if args.run_judger and config.get("judger") != args.run_judger:
                    continue
                
                filtered_placeholders.append(p)
            
            # 应用数量限制
            if args.run_limit and len(filtered_placeholders) > args.run_limit:
                filtered_placeholders = filtered_placeholders[:args.run_limit]
            
            if not filtered_placeholders:
                print("✅ 没有匹配过滤条件的占位符实验")
                return
            
            placeholder_files = [
                str(placeholder_manager.placeholders_dir / p["filename"])
                for p in filtered_placeholders
            ]
            
            workers = getattr(args, 'workers', 1)
            print(f"🚀 开始执行 {len(placeholder_files)} 个待执行实验...")
            
            if args.run_model:
                print(f"   📱 模型过滤: {args.run_model}")
            if args.run_attack:
                print(f"   ⚔️ 攻击过滤: {args.run_attack}")
            if args.run_defense:
                print(f"   🛡️ 防御过滤: {args.run_defense}")
            if args.run_dataset:
                print(f"   📊 数据集过滤: {args.run_dataset}")
            if args.run_judger:
                print(f"   👨‍⚖️ 评判器过滤: {args.run_judger}")
            if args.run_limit:
                print(f"   🔢 数量限制: {args.run_limit}")
            if hasattr(args, 'sample_limit') and args.sample_limit:
                print(f"   📊 样本限制: {args.sample_limit}")
            if args.rerun_failed:
                print("   🔁 非success样本重跑: 启用")
            if args.force_rerun:
                print("   🔁 强制重跑: 启用")
            if args.verbose:
                print(f"   📋 详细输出: 启用 (最大长度: {args.max_length})")
            print(f"   提示: 使用 --verbose 查看详细的实验过程和结果")
            
            # 传递sample_limit参数（如果存在）
            sample_limit = getattr(args, 'sample_limit', None)
            results = runner.run_batch_placeholders(placeholder_files, workers, sample_limit)
            
            # 统计结果
            completed = len([r for r in results if r.get("status") == "completed"])
            success = len([r for r in results if is_success_result(r)])
            failed = len([r for r in results if r.get("status") == "failed"])
            skipped = len([r for r in results if r.get("status") == "skipped"])
            
            print(f"\n📊 执行完成:")
            print(f"  ✅ 成功: {success}")
            print(f"  ✅ 完成: {completed}")
            print(f"  ❌ 失败: {failed}")
            print(f"  ⏭️ 跳过: {skipped}")
            print(f"  📁 总计: {len(results)}")
            return
        
        # 统一占位符模式：所有实验都通过占位符系统执行
        
        # 处理快捷预设
        preset_used = False
        if args.baseline:
            args.attack = args.attack or ["no_attack"]
            args.defense = args.defense or ["no_defense"]
            preset_used = True
            print("🎯 使用基线评估预设: no_attack + no_defense")
        elif args.security_test:
            args.attack = args.attack or ["ArtPrompt"]
            args.defense = args.defense or ["no_defense"]
            preset_used = True
            print("⚔️ 使用安全测试预设: ArtPrompt + no_defense")
        elif args.defense_eval:
            args.attack = args.attack or ["GPTFUZZER"]
            args.defense = args.defense or ["smooth_llm"]
            preset_used = True
            print("🛡️ 使用防御评估预设: GPTFUZZER + smooth_llm")
        
        # 处理恢复模式
        if args.resume:
            from experiments.core.placeholder_system import ExperimentPlaceholder
            from experiments.core.placeholder_runner import PlaceholderExperimentRunner
            placeholder_manager = ExperimentPlaceholder(seed=getattr(args, 'seed', 42))
            failed_placeholders = placeholder_manager.list_placeholders("failed")
            
            if not failed_placeholders:
                print("✅ 没有失败的实验需要恢复")
                return
            
            print(f"🔄 恢复模式: 发现 {len(failed_placeholders)} 个失败的实验")
            placeholder_files = [
                str(placeholder_manager.placeholders_dir / p["filename"])
                for p in failed_placeholders
            ]
            
            from experiments.core.placeholder_runner import PlaceholderExperimentRunner
            runner = PlaceholderExperimentRunner(
                verbose=args.verbose,
                max_length=args.max_length,
                seed=getattr(args, 'seed', 42),
                rerun_failed=args.rerun_failed,
                force_rerun=args.force_rerun
            )
            results = runner.run_batch_placeholders(placeholder_files, workers=getattr(args, 'workers', 1))
            
            completed = len([r for r in results if r.get("status") == "completed"])
            print(f"🔄 恢复完成: {completed}/{len(results)} 个实验成功")
            return
        
        # 如果没有提供任何参数，显示帮助
        if not any([args.model, args.attack, args.defense, args.config, args.phase, 
                   args.list_placeholders, args.run_placeholders, preset_used, args.resume]):
            parser.print_help()
            return
        
        # 5要素智能默认值与自动占位符创建
        if any([args.model, args.attack, args.defense, args.dataset, args.judger]) or args.config or args.phase or preset_used:
            # 智能默认值：自动补全未提供的参数
            models = args.model or ["gpt-4o"]
            attacks = args.attack or ["no_attack"]
            defenses = args.defense or ["no_defense"]
            datasets = args.dataset or ["harmbench"]
            judgers = args.judger or ["harmbench_judger"]
            
            # 显示自动补全的参数
            if not args.model and not preset_used:
                print("📝 自动默认值: model=gpt-4o")
            if not args.attack and not preset_used:
                print("📝 自动默认值: attack=no_attack")
            if not args.defense and not preset_used:
                print("📝 自动默认值: defense=no_defense")
            if not args.dataset:
                print("📝 自动默认值: dataset=harmbench")
            if not args.judger:
                print("📝 自动默认值: judger=harmbench_judger")
            
            # 处理特殊实验类型
            if args.phase:
                print(f"🎯 Phase {args.phase} 实验模式")
                
                if args.phase == 1:
                    # Phase1强制使用占位符系统
                    print("📋 Phase 1实验强制使用占位符系统执行")
                    from experiments.core.placeholder_system import ExperimentPlaceholder
                    from experiments.core.placeholder_runner import PlaceholderExperimentRunner
                    placeholder_manager = ExperimentPlaceholder(seed=args.seed)
                    
                    # 生成占位符文件
                    placeholder_files = placeholder_manager.generate_phase1_placeholders(args.sample_limit)
                    print(f"📝 已生成 {len(placeholder_files)} 个Phase1占位符文件")
                    
                    # 显示生成的文件列表（用于调试）
                    if args.verbose:
                        for i, pf in enumerate(placeholder_files[:3]):  # 只显示前3个
                            print(f"   {i+1}. {Path(pf).name}")
                        if len(placeholder_files) > 3:
                            print(f"   ... 以及另外 {len(placeholder_files)-3} 个文件")
                    
                    if args.generate_only:
                        print("✅ Phase1占位符生成完成，使用以下命令执行:")
                        print("   python -m experiments --run-placeholders")
                        return
                    
                    # 执行Phase1实验
                    print("🚀 开始通过占位符系统执行Phase1实验...")
                    runner = PlaceholderExperimentRunner(
                        verbose=args.verbose,
                        max_length=args.max_length,
                        seed=args.seed,
                        rerun_failed=args.rerun_failed,
                        force_rerun=args.force_rerun
                    )
                    
                    # 添加调试信息
                    if args.verbose:
                        print(f"   📊 样本限制: {args.sample_limit}")
                        print(f"   🎲 随机种子: {args.seed}")
                        print(f"   👥 工作进程: {getattr(args, 'workers', 1)}")
                    
                    # 执行占位符实验
                    results = runner.run_batch_placeholders(placeholder_files, workers=getattr(args, 'workers', 1), sample_limit=args.sample_limit)
                    
                    # 统计结果
                    completed = len([r for r in results if r.get("status") == "completed"])
                    failed = len([r for r in results if r.get("status") == "failed"])
                    skipped = len([r for r in results if r.get("status") == "skipped"])
                    
                    print(f"✅ Phase1实验执行完成:")
                    print(f"   ✅ 完成: {completed}")
                    print(f"   ❌ 失败: {failed}")
                    print(f"   ⏭️ 跳过: {skipped}")
                    print(f"   📁 总计: {len(results)}")
                    print("📊 结果已保存在占位符系统中，使用 --dashboard 查看详细结果")
                    return
                elif args.phase == 4:
                    # Phase4强制使用占位符系统
                    print("📋 Phase 4实验强制使用占位符系统执行")
                    from experiments.core.placeholder_system import ExperimentPlaceholder
                    from experiments.core.placeholder_runner import PlaceholderExperimentRunner
                    placeholder_manager = ExperimentPlaceholder(seed=args.seed)
                    
                    # 生成占位符文件
                    placeholder_files = placeholder_manager.generate_phase4_placeholders(args.sample_limit)
                    print(f"📝 已生成 {len(placeholder_files)} 个Phase4占位符文件")
                    
                    # 显示生成的文件列表（用于调试）
                    if args.verbose:
                        for i, pf in enumerate(placeholder_files[:3]):  # 只显示前3个
                            print(f"   {i+1}. {Path(pf).name}")
                        if len(placeholder_files) > 3:
                            print(f"   ... 以及另外 {len(placeholder_files)-3} 个文件")
                    
                    if args.generate_only:
                        print("✅ Phase4占位符生成完成，使用以下命令执行:")
                        print("   python -m experiments --run-placeholders")
                        return
                    
                    # 执行Phase4实验
                    print("🚀 开始通过占位符系统执行Phase4实验...")
                    runner = PlaceholderExperimentRunner(
                        verbose=args.verbose,
                        max_length=args.max_length,
                        seed=args.seed,
                        rerun_failed=args.rerun_failed,
                        force_rerun=args.force_rerun
                    )
                    
                    # 添加调试信息
                    if args.verbose:
                        print(f"   📊 样本限制: {args.sample_limit}")
                        print(f"   🎲 随机种子: {args.seed}")
                        print(f"   👥 工作进程: {getattr(args, 'workers', 1)}")
                        print(f"   💡 注意: Phase4实验量大，建议使用 --generate-only 先生成占位符")
                    
                    # 执行占位符实验
                    results = runner.run_batch_placeholders(placeholder_files, workers=getattr(args, 'workers', 1), sample_limit=args.sample_limit)
                    
                    # 统计结果
                    completed = len([r for r in results if r.get("status") == "completed"])
                    failed = len([r for r in results if r.get("status") == "failed"])
                    skipped = len([r for r in results if r.get("status") == "skipped"])
                    
                    print(f"✅ Phase4实验执行完成:")
                    print(f"   ✅ 完成: {completed}")
                    print(f"   ❌ 失败: {failed}")
                    print(f"   ⏭️ 跳过: {skipped}")
                    print(f"   📁 总计: {len(results)}")
                    print("📊 结果已保存在占位符系统中，使用 --dashboard 查看详细结果")
                    return
                else:
                    # 其他Phase使用原有方式
                    interface = PromptSecurityInterface()
                    result = interface.run_phase_experiment(args.phase, sample_limit=args.sample_limit, seed=args.seed)
                    print("✅ Phase实验完成!")
                    if "selected_models" in result:
                        print(f"选择的模型: {result['selected_models']}")
                    return
            
            # 处理配置文件
            if args.config:
                interface = PromptSecurityInterface()
                result = interface.run_experiment(config=args.config)
                print("📄 配置文件实验已通过占位符系统执行")
                return
            
            # 标准占位符模式
            from experiments.core.placeholder_system import ExperimentPlaceholder
            from experiments.core.placeholder_runner import PlaceholderExperimentRunner
            placeholder_manager = ExperimentPlaceholder(seed=args.seed)
            
            # 检测是否为批量模式（多个参数值）
            is_batch = any(len(param) > 1 for param in [models, attacks, defenses, datasets, judgers])
            
            if is_batch:
                print(f"🔄 批量模式：{len(models)}×{len(attacks)}×{len(defenses)}×{len(datasets)}×{len(judgers)} = {len(models)*len(attacks)*len(defenses)*len(datasets)*len(judgers)} 个实验")
            else:
                print(f"🎯 单实验模式: {models[0]} + {attacks[0]} + {defenses[0]} + {datasets[0]} + {judgers[0]}")
            
            # 生成占位符
            placeholder_files = placeholder_manager.generate_batch_placeholders(
                models=models,
                attacks=attacks,
                defenses=defenses,
                datasets=datasets,
                judgers=judgers,
                sample_limit=args.sample_limit,
                seed=args.seed,
                multi_judger=getattr(args, 'multi_judger', False) and len(judgers) > 1
            )
            
            if not placeholder_files:
                print("❌ 没有生成有效的占位符文件")
                sys.exit(1)
            
            print(f"📝 已生成 {len(placeholder_files)} 个占位符文件")
            
            # 仅生成模式
            if args.generate_only:
                print("✅ 占位符生成完成，使用以下命令执行:")
                print("   python -m experiments --run-placeholders")
                print("   python -m experiments --run-placeholders --verbose")
                return
            
            # 执行占位符实验
            runner = PlaceholderExperimentRunner(
                verbose=args.verbose,
                max_length=args.max_length,
                seed=args.seed,
                rerun_failed=args.rerun_failed,
                force_rerun=args.force_rerun
            )
            workers = getattr(args, 'workers', 1)
            
            print("🚀 开始执行实验...")
            results = runner.run_batch_placeholders(placeholder_files, workers)
            
            # 结果汇总
            completed = len([r for r in results if r.get("status") == "completed"])
            success = len([r for r in results if is_success_result(r)])
            failed = len([r for r in results if r.get("status") == "failed"])
            skipped = len([r for r in results if r.get("status") == "skipped"])
            
            print(f"\n📊 执行完成: ✅{success} ✅完成{completed} ❌{failed} ⏭️{skipped} | 📁{len(results)}")
            print("💾 结果保存在: experiments/placeholders/")
            print("🔍 使用 --dashboard 查看详细结果")
            
            return
    
    except KeyboardInterrupt:
        print("\n⚠️ 实验被用户中断")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ 实验执行失败: {e}")
        import traceback
        print("\n错误详情:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
