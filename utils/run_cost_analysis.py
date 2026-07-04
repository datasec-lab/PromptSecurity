#!/usr/bin/env python3
"""
Run Cost Analysis on Existing Configurations
===========================================

This script runs cost analysis on existing batch configurations
and generates reports for different scenarios.
"""

import os
import sys
import glob
from pathlib import Path
from cost_analysis import CostAnalyzer
from rich.console import Console
from rich.panel import Panel

def main():
    """Run cost analysis on various configurations"""
    console = Console()
    analyzer = CostAnalyzer()
    
    # Base path for configurations
    base_path = Path(".")
    
    console.print(Panel(
        "[bold cyan]PromptSecurity Cost Analysis Report[/bold cyan]\n"
        "This analysis provides cost estimates for running experiments\n"
        "across different models, datasets, attacks, and defenses.",
        title="💰 Cost Analysis"
    ))
    
    # Scenario 1: Quick evaluation with budget models
    console.print("\n[bold yellow]Scenario 1: Budget-Friendly Quick Evaluation[/bold yellow]")
    budget_models = ["gpt-4o-mini", "claude-3-5-haiku-20241022", "gemini-1.5-flash", "deepseek-chat"]
    quick_attacks = ["None", "ArtPrompt", "FlipAttack", "PastTense"]
    basic_defenses = ["None", "InputFilter", "OutputFilter"]
    
    df1 = analyzer.generate_cost_report(
        models=budget_models,
        datasets=["harmbench"],
        attacks=quick_attacks,
        defenses=basic_defenses,
        prompt_count=50
    )
    
    console.print(f"[green]Total cost for budget scenario: ${df1['Total Cost'].sum():.2f}[/green]")
    analyzer.save_detailed_report(df1, "cost_analysis_budget.json")
    
    # Scenario 2: Comprehensive evaluation with premium models
    console.print("\n[bold yellow]Scenario 2: Comprehensive Premium Evaluation[/bold yellow]")
    premium_models = ["gpt-4o", "claude-3-5-sonnet-latest", "gemini-1.5-pro", "o1-mini"]
    comprehensive_attacks = ["None", "ReNeLLM", "TapAttack", "PAIR", "GPTFUZZER", "ArtPrompt"]
    comprehensive_defenses = ["None", "JailGuard", "BackTranslation", "SmoothLLM", "RPO"]
    
    df2 = analyzer.generate_cost_report(
        models=premium_models,
        datasets=["harmbench", "jbb"],
        attacks=comprehensive_attacks,
        defenses=comprehensive_defenses,
        prompt_count=100
    )
    
    console.print(f"[green]Total cost for comprehensive scenario: ${df2['Total Cost'].sum():.2f}[/green]")
    analyzer.save_detailed_report(df2, "cost_analysis_comprehensive.json")
    
    # Scenario 3: Research-grade evaluation
    console.print("\n[bold yellow]Scenario 3: Research-Grade Full Evaluation[/bold yellow]")
    research_models = ["gpt-4o", "claude-3-5-sonnet-latest", "gemini-1.5-pro", "o1", "deepseek-reasoner"]
    all_attacks = [
        "None", "ReNeLLM", "PAIR", "TapAttack", "ABJAttack", "DRA", "GPTFUZZER", 
        "ArtPrompt", "CodeAttack", "CodeChameleon", "FlipAttack", "InceptionAttack",
        "MultilingualJailbreak", "PastTense", "PersuasiveInContext", "DrAttack", "IFSJAttack"
    ]
    all_defenses = [
        "None", "JailGuard", "PerplexityFilter", "BackTranslation", "SmoothLLM", 
        "RPO", "GradSafe", "PrimeGuard", "InputFilter", "OutputFilter"
    ]
    
    # For research grade, we'll use a smaller sample to keep costs manageable
    df3 = analyzer.generate_cost_report(
        models=research_models[:3],  # Top 3 models
        datasets=["harmbench", "jbb", "airbench"],
        attacks=all_attacks[:8],  # Top 8 attacks
        defenses=all_defenses[:5],  # Top 5 defenses
        prompt_count=200
    )
    
    console.print(f"[green]Total cost for research scenario: ${df3['Total Cost'].sum():.2f}[/green]")
    analyzer.save_detailed_report(df3, "cost_analysis_research.json")
    
    # Analyze existing batch configurations if they exist
    batch_configs = glob.glob("experiments/batch_configs/*.json")
    if batch_configs:
        console.print("\n[bold yellow]Analyzing Existing Batch Configurations[/bold yellow]")
        
        for config_path in batch_configs[:3]:  # Analyze first 3 configs
            config_name = Path(config_path).stem
            console.print(f"\n[cyan]Analyzing {config_name}...[/cyan]")
            
            try:
                results = analyzer.analyze_batch_config(config_path)
                if results:
                    total_cost = sum(r.total_cost for r in results)
                    total_time = sum(r.estimated_time_minutes for r in results)
                    console.print(f"[green]• Cost: ${total_cost:.2f}[/green]")
                    console.print(f"[blue]• Time: {total_time:.1f} minutes[/blue]")
                    console.print(f"[yellow]• Experiments: {len(results)}[/yellow]")
                    
                    # Save individual config analysis
                    output_file = f"cost_analysis_{config_name}.json"
                    df_config = analyzer.generate_cost_report(
                        models=[r.model_name for r in results[:1]],
                        datasets=[r.dataset_name for r in results[:1]],
                        attacks=[r.attack_name for r in results[:5]],
                        defenses=[r.defense_name for r in results[:5]],
                        prompt_count=results[0].prompt_count if results else 100
                    )
                    analyzer.save_detailed_report(df_config, output_file)
                else:
                    console.print("[red]• No valid experiments found[/red]")
            except Exception as e:
                console.print(f"[red]• Error analyzing {config_name}: {e}[/red]")
    
    # Print cost comparison table
    console.print("\n[bold yellow]Cost Comparison Summary[/bold yellow]")
    from rich.table import Table
    
    summary_table = Table(title="Scenario Cost Summary")
    summary_table.add_column("Scenario", style="cyan")
    summary_table.add_column("Total Cost", style="green")
    summary_table.add_column("Experiments", style="yellow")
    summary_table.add_column("Avg Cost/Exp", style="blue")
    
    scenarios = [
        ("Budget Quick", df1, "50 prompts, 4 models, 4 attacks, 3 defenses"),
        ("Comprehensive", df2, "100 prompts, 4 models, 6 attacks, 5 defenses"),
        ("Research Grade", df3, "200 prompts, 3 models, 8 attacks, 5 defenses")
    ]
    
    for name, df, desc in scenarios:
        total_cost = df['Total Cost'].sum()
        num_experiments = len(df)
        avg_cost = total_cost / num_experiments if num_experiments > 0 else 0
        
        summary_table.add_row(
            name,
            f"${total_cost:.2f}",
            str(num_experiments),
            f"${avg_cost:.3f}"
        )
    
    console.print(summary_table)
    
    # Print general recommendations
    console.print(Panel(
        "[bold]💡 General Recommendations:[/bold]\n\n"
        "🔸 [green]Budget-conscious:[/green] Use mini/nano models + single-turn attacks\n"
        "🔸 [yellow]Balanced approach:[/yellow] Mix premium and budget models based on attack complexity\n"
        "🔸 [red]Research-grade:[/red] Use premium models for complex multi-turn attacks\n\n"
        "🔸 [blue]Cost optimization:[/blue]\n"
        "   • Start with small sample sizes (20-50 prompts)\n"
        "   • Use batch APIs for 50% discount\n"
        "   • Enable prompt caching for repetitive queries\n"
        "   • Consider DeepSeek models for budget-friendly alternatives\n\n"
        "🔸 [magenta]Time optimization:[/magenta]\n"
        "   • Parallelize experiments where possible\n"
        "   • Use faster models for initial screening\n"
        "   • Multi-turn attacks take significantly longer",
        title="📊 Recommendations"
    ))
    
    console.print(f"\n[bold green]Analysis complete! Reports saved to cost_analysis_*.json files[/bold green]")

if __name__ == "__main__":
    main()