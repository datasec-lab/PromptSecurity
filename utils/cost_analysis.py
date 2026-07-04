#!/usr/bin/env python3
"""
Comprehensive Cost Analysis Script for Cross-Experiment Evaluation
================================================================

This script analyzes the cost of running experiments across different models,
datasets, attacks, and defenses in the PromptSecurity framework.

It provides detailed cost breakdowns and optimization suggestions.
"""

import json
import os
import glob
import argparse
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
import pandas as pd

@dataclass
class ModelPricing:
    """Pricing information for different models"""
    input_price_per_million: float
    output_price_per_million: float
    currency: str = "USD"
    notes: str = ""

@dataclass
class DatasetInfo:
    """Information about datasets"""
    name: str
    total_samples: int
    typical_sample_size: int
    default_sample_size: int = 100
    
@dataclass
class AttackInfo:
    """Information about attacks"""
    name: str
    avg_queries_per_prompt: int
    avg_input_tokens_per_query: int
    avg_output_tokens_per_query: int
    uses_additional_models: bool = False
    additional_model_queries: int = 0

@dataclass
class DefenseInfo:
    """Information about defenses"""
    name: str
    additional_processing_cost: float = 0.0
    tokens_multiplier: float = 1.0
    uses_additional_models: bool = False
    additional_model_queries: int = 0

@dataclass
class CostBreakdown:
    """Detailed cost breakdown for an experiment"""
    model_name: str
    dataset_name: str
    attack_name: str
    defense_name: str
    prompt_count: int
    base_cost: float
    attack_cost: float
    defense_cost: float
    judger_cost: float
    total_cost: float
    estimated_time_minutes: float
    
class CostAnalyzer:
    """Main cost analysis class"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.console = Console()
        
        # Initialize pricing data
        self.model_pricing = self._initialize_model_pricing()
        self.datasets = self._initialize_dataset_info()
        self.attacks = self._initialize_attack_info()
        self.defenses = self._initialize_defense_info()
        
        # GPU costs (per hour)
        self.gpu_costs = {
            "RTX 4090": 0.50,
            "A100": 2.00,
            "H100": 4.00,
            "A6000": 1.50,
            "V100": 1.00,
        }
        
    def _initialize_model_pricing(self) -> Dict[str, ModelPricing]:
        """Initialize pricing for different API models (2025 pricing)"""
        return {
            # OpenAI Models
            "gpt-4o": ModelPricing(3.0, 10.0, notes="Latest GPT-4 Omni model"),
            "gpt-4o-mini": ModelPricing(0.15, 0.60, notes="Budget GPT-4 variant"),
            "gpt-4.1": ModelPricing(2.5, 8.0, notes="26% cheaper than GPT-4o"),
            "gpt-4.1-mini": ModelPricing(0.10, 0.40, notes="Fastest budget option"),
            "gpt-4.1-nano": ModelPricing(0.05, 0.20, notes="Cheapest and fastest"),
            "gpt-4-turbo": ModelPricing(10.0, 30.0, notes="Legacy pricing"),
            "gpt-3.5-turbo": ModelPricing(0.5, 1.5, notes="Legacy model"),
            "o1": ModelPricing(15.0, 60.0, notes="Reasoning model"),
            "o1-mini": ModelPricing(3.0, 12.0, notes="Smaller reasoning model"),
            "o1-preview": ModelPricing(15.0, 60.0, notes="Preview reasoning model"),
            
            # Claude Models
            "claude-3-5-sonnet-latest": ModelPricing(3.0, 15.0, notes="Latest Claude 3.5"),
            "claude-3-5-sonnet-20241022": ModelPricing(3.0, 15.0, notes="Claude 3.5 Sonnet"),
            "claude-3-5-sonnet-20240620": ModelPricing(3.0, 15.0, notes="Claude 3.5 Sonnet"),
            "claude-3-5-haiku-20241022": ModelPricing(1.0, 5.0, notes="Claude 3.5 Haiku"),
            "claude-3-opus-20240229": ModelPricing(15.0, 75.0, notes="Claude 3 Opus"),
            "claude-3-opus-latest": ModelPricing(15.0, 75.0, notes="Latest Claude 3 Opus"),
            "claude-3-sonnet-20240229": ModelPricing(3.0, 15.0, notes="Claude 3 Sonnet"),
            "claude-3-haiku-20240307": ModelPricing(0.25, 1.25, notes="Claude 3 Haiku"),
            "claude-4": ModelPricing(3.0, 15.0, notes="Estimated Claude 4 pricing"),
            
            # Gemini Models
            "gemini-1.5-pro": ModelPricing(3.5, 10.5, notes="Gemini 1.5 Pro"),
            "gemini-1.5-pro-latest": ModelPricing(3.5, 10.5, notes="Latest Gemini 1.5 Pro"),
            "gemini-1.5-flash": ModelPricing(0.075, 0.30, notes="Gemini 1.5 Flash"),
            "gemini-2.0-flash-exp": ModelPricing(0.0, 0.0, notes="Free during experimental period"),
            "gemini-2.0-flash-thinking-exp": ModelPricing(0.0, 0.0, notes="Free during experimental period"),
            "gemini-2.5-pro": ModelPricing(1.25, 10.0, notes="Up to 200K tokens"),
            "gemini-2.5-flash": ModelPricing(0.10, 0.40, notes="Gemini 2.5 Flash"),
            
            # DeepSeek Models
            "deepseek-chat": ModelPricing(0.27, 1.10, notes="DeepSeek V3 chat model"),
            "deepseek-reasoner": ModelPricing(0.55, 2.19, notes="DeepSeek R1 reasoning model"),
            
            # DeepInfra Models (estimated based on model size)
            "meta-llama-3.1-405b-instruct": ModelPricing(2.7, 2.7, notes="Via DeepInfra"),
            "meta-llama-3.1-70b-instruct": ModelPricing(0.59, 0.79, notes="Via DeepInfra"),
            "meta-llama-3.1-8b-instruct": ModelPricing(0.08, 0.08, notes="Via DeepInfra"),
            "qwen-2.5-72b-instruct": ModelPricing(0.59, 0.79, notes="Via DeepInfra"),
            "qwen-2.5-7b-instruct": ModelPricing(0.07, 0.07, notes="Via DeepInfra"),
            "mixtral-8x22b-instruct": ModelPricing(0.65, 0.65, notes="Via DeepInfra"),
            "mixtral-8x7b-instruct": ModelPricing(0.24, 0.24, notes="Via DeepInfra"),
        }
    
    def _initialize_dataset_info(self) -> Dict[str, DatasetInfo]:
        """Initialize dataset information"""
        return {
            "harmbench": DatasetInfo(
                name="HarmBench",
                total_samples=1528,  # Based on the CSV file we found
                typical_sample_size=100,
                default_sample_size=100
            ),
            "jbb": DatasetInfo(
                name="JBB (Jailbreak Bench)",
                total_samples=520,  # Estimated from JBB-Behaviors dataset
                typical_sample_size=100,
                default_sample_size=100
            ),
            "airbench": DatasetInfo(
                name="AIR-Bench",
                total_samples=1000,  # Estimated from AIR-Bench-2024
                typical_sample_size=100,
                default_sample_size=100
            ),
            "combined": DatasetInfo(
                name="Combined Dataset",
                total_samples=3048,  # Sum of all datasets
                typical_sample_size=200,
                default_sample_size=200
            )
        }
    
    def _initialize_attack_info(self) -> Dict[str, AttackInfo]:
        """Initialize attack information based on typical usage patterns"""
        return {
            "None": AttackInfo("No Attack", 1, 100, 150, False, 0),
            
            # Single-turn attacks
            "ArtPrompt": AttackInfo("ArtPrompt", 1, 500, 200, False, 0),
            "CodeAttack": AttackInfo("CodeAttack", 1, 300, 200, False, 0),
            "CodeChameleon": AttackInfo("CodeChameleon", 1, 400, 250, False, 0),
            "FlipAttack": AttackInfo("FlipAttack", 1, 150, 180, False, 0),
            "InceptionAttack": AttackInfo("InceptionAttack", 1, 200, 200, False, 0),
            "MultilingualJailbreak": AttackInfo("MultilingualJailbreak", 2, 200, 200, False, 0),
            "PastTense": AttackInfo("PastTense", 1, 120, 150, False, 0),
            "DrAttack": AttackInfo("DrAttack", 1, 180, 200, False, 0),
            "IFSJAttack": AttackInfo("IFSJAttack", 1, 200, 180, False, 0),
            "PersuasiveInContext": AttackInfo("PersuasiveInContext", 1, 300, 250, False, 0),
            
            # Multi-turn attacks
            "ReNeLLM": AttackInfo("ReNeLLM", 10, 250, 200, True, 10),
            "PAIR": AttackInfo("PAIR", 15, 300, 250, True, 15),
            "TapAttack": AttackInfo("TapAttack", 25, 350, 300, True, 25),
            "ABJAttack": AttackInfo("ABJAttack", 8, 200, 180, True, 8),
            "DRA": AttackInfo("DRA", 5, 180, 150, True, 5),
            "GPTFUZZER": AttackInfo("GPTFUZZER", 20, 300, 250, True, 20),
        }
    
    def _initialize_defense_info(self) -> Dict[str, DefenseInfo]:
        """Initialize defense information"""
        return {
            "None": DefenseInfo("No Defense", 0.0, 1.0, False, 0),
            "JailGuard": DefenseInfo("JailGuard", 0.002, 1.1, True, 1),
            "PerplexityFilter": DefenseInfo("PerplexityFilter", 0.001, 1.05, True, 1),
            "BackTranslation": DefenseInfo("BackTranslation", 0.005, 1.5, True, 2),
            "SmoothLLM": DefenseInfo("SmoothLLM", 0.003, 1.2, False, 0),
            "RPO": DefenseInfo("RPO", 0.004, 1.3, True, 1),
            "GradSafe": DefenseInfo("GradSafe", 0.002, 1.1, True, 1),
            "PrimeGuard": DefenseInfo("PrimeGuard", 0.006, 1.4, True, 2),
            "InputFilter": DefenseInfo("InputFilter", 0.001, 1.05, True, 1),
            "OutputFilter": DefenseInfo("OutputFilter", 0.001, 1.05, True, 1),
        }
    
    def calculate_experiment_cost(self, 
                                model_name: str,
                                dataset_name: str,
                                attack_name: str = "None",
                                defense_name: str = "None",
                                prompt_count: Optional[int] = None,
                                judger_model: str = "gpt-4o-mini") -> CostBreakdown:
        """Calculate cost for a specific experiment configuration"""
        
        # Get model pricing
        model_pricing = self.model_pricing.get(model_name)
        if not model_pricing:
            # Try to find similar model or use default
            model_pricing = self._get_similar_model_pricing(model_name)
        
        # Get dataset info
        dataset = self.datasets.get(dataset_name, self.datasets["harmbench"])
        if prompt_count is None:
            prompt_count = dataset.typical_sample_size
        
        # Get attack and defense info
        attack = self.attacks.get(attack_name, self.attacks["None"])
        defense = self.defenses.get(defense_name, self.defenses["None"])
        
        # Calculate base cost (without attacks/defenses)
        base_input_tokens = 100 * prompt_count  # Base prompt tokens
        base_output_tokens = 150 * prompt_count  # Base response tokens
        
        base_cost = (
            (base_input_tokens * model_pricing.input_price_per_million / 1_000_000) +
            (base_output_tokens * model_pricing.output_price_per_million / 1_000_000)
        )
        
        # Calculate attack cost
        attack_input_tokens = attack.avg_input_tokens_per_query * attack.avg_queries_per_prompt * prompt_count
        attack_output_tokens = attack.avg_output_tokens_per_query * attack.avg_queries_per_prompt * prompt_count
        
        attack_cost = (
            (attack_input_tokens * model_pricing.input_price_per_million / 1_000_000) +
            (attack_output_tokens * model_pricing.output_price_per_million / 1_000_000)
        )
        
        # Add additional model costs for attacks
        if attack.uses_additional_models:
            # Assume additional models use same pricing as main model
            additional_attack_cost = (
                (attack.additional_model_queries * 200 * model_pricing.input_price_per_million / 1_000_000) +
                (attack.additional_model_queries * 150 * model_pricing.output_price_per_million / 1_000_000)
            ) * prompt_count
            attack_cost += additional_attack_cost
        
        # Calculate defense cost
        defense_multiplier = defense.tokens_multiplier
        defense_cost = defense.additional_processing_cost * prompt_count
        
        # Add additional model costs for defenses
        if defense.uses_additional_models:
            judger_pricing = self.model_pricing.get(judger_model, model_pricing)
            additional_defense_cost = (
                (defense.additional_model_queries * 150 * judger_pricing.input_price_per_million / 1_000_000) +
                (defense.additional_model_queries * 50 * judger_pricing.output_price_per_million / 1_000_000)
            ) * prompt_count
            defense_cost += additional_defense_cost
        
        # Calculate judger cost
        judger_pricing = self.model_pricing.get(judger_model, model_pricing)
        judger_input_tokens = 300 * prompt_count  # Judger prompt + response to evaluate
        judger_output_tokens = 50 * prompt_count   # Judger response (binary yes/no typically)
        
        judger_cost = (
            (judger_input_tokens * judger_pricing.input_price_per_million / 1_000_000) +
            (judger_output_tokens * judger_pricing.output_price_per_million / 1_000_000)
        )
        
        # Apply defense multiplier to main costs
        total_main_cost = (base_cost + attack_cost) * defense_multiplier
        total_cost = total_main_cost + defense_cost + judger_cost
        
        # Estimate time (rough approximation)
        estimated_time = (
            (attack.avg_queries_per_prompt * prompt_count * 2) +  # 2 seconds per query
            (defense.additional_model_queries * prompt_count * 1.5) +  # 1.5 seconds per defense query
            (prompt_count * 1)  # 1 second per judger query
        ) / 60  # Convert to minutes
        
        return CostBreakdown(
            model_name=model_name,
            dataset_name=dataset_name,
            attack_name=attack_name,
            defense_name=defense_name,
            prompt_count=prompt_count,
            base_cost=base_cost,
            attack_cost=attack_cost,
            defense_cost=defense_cost,
            judger_cost=judger_cost,
            total_cost=total_cost,
            estimated_time_minutes=estimated_time
        )
    
    def _get_similar_model_pricing(self, model_name: str) -> ModelPricing:
        """Get pricing for similar model if exact match not found"""
        model_name_lower = model_name.lower()
        
        # Try to match by model family
        if "gpt-4" in model_name_lower:
            if "mini" in model_name_lower or "nano" in model_name_lower:
                return self.model_pricing["gpt-4o-mini"]
            else:
                return self.model_pricing["gpt-4o"]
        elif "claude" in model_name_lower:
            if "haiku" in model_name_lower:
                return self.model_pricing["claude-3-5-haiku-20241022"]
            elif "opus" in model_name_lower:
                return self.model_pricing["claude-3-opus-latest"]
            else:
                return self.model_pricing["claude-3-5-sonnet-latest"]
        elif "gemini" in model_name_lower:
            if "flash" in model_name_lower:
                return self.model_pricing["gemini-1.5-flash"]
            else:
                return self.model_pricing["gemini-1.5-pro"]
        elif "deepseek" in model_name_lower:
            if "reasoner" in model_name_lower:
                return self.model_pricing["deepseek-reasoner"]
            else:
                return self.model_pricing["deepseek-chat"]
        elif "llama" in model_name_lower:
            if "405b" in model_name_lower:
                return self.model_pricing["meta-llama-3.1-405b-instruct"]
            elif "70b" in model_name_lower:
                return self.model_pricing["meta-llama-3.1-70b-instruct"]
            else:
                return self.model_pricing["meta-llama-3.1-8b-instruct"]
        
        # Default to mid-range pricing
        return ModelPricing(1.0, 3.0, notes="Estimated pricing")
    
    def analyze_batch_config(self, config_path: str) -> List[CostBreakdown]:
        """Analyze cost for a batch configuration file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        results = []
        
        # Extract model info
        target_model = config.get("target_model", {})
        if isinstance(target_model, str):
            # Model config file path
            model_name = Path(target_model).stem
        else:
            # Inline model config
            model_name = target_model.get("model_name", "gpt-4o-mini")
        
        # Extract dataset info
        test_data = config.get("test_data_source", {})
        dataset_name = test_data.get("type", "harmbench")
        prompt_count = test_data.get("sample_size", 100)
        
        # Get attacks and defenses
        attacks = config.get("attacks", [{"name": "None"}])
        defenses = config.get("defenses", [{"name": "None"}])
        
        # Calculate costs for all combinations
        for attack in attacks:
            attack_name = attack.get("name", "None")
            for defense in defenses:
                defense_name = defense.get("name", "None")
                
                cost_breakdown = self.calculate_experiment_cost(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    attack_name=attack_name,
                    defense_name=defense_name,
                    prompt_count=prompt_count
                )
                results.append(cost_breakdown)
        
        return results
    
    def generate_cost_report(self, 
                           models: List[str],
                           datasets: List[str],
                           attacks: List[str],
                           defenses: List[str],
                           prompt_count: int = 100) -> pd.DataFrame:
        """Generate comprehensive cost report"""
        results = []
        
        with Progress() as progress:
            task = progress.add_task("Calculating costs...", total=len(models)*len(datasets)*len(attacks)*len(defenses))
            
            for model in models:
                for dataset in datasets:
                    for attack in attacks:
                        for defense in defenses:
                            cost_breakdown = self.calculate_experiment_cost(
                                model_name=model,
                                dataset_name=dataset,
                                attack_name=attack,
                                defense_name=defense,
                                prompt_count=prompt_count
                            )
                            results.append(cost_breakdown)
                            progress.advance(task)
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                "Model": r.model_name,
                "Dataset": r.dataset_name,
                "Attack": r.attack_name,
                "Defense": r.defense_name,
                "Prompts": r.prompt_count,
                "Base Cost": r.base_cost,
                "Attack Cost": r.attack_cost,
                "Defense Cost": r.defense_cost,
                "Judger Cost": r.judger_cost,
                "Total Cost": r.total_cost,
                "Time (min)": r.estimated_time_minutes
            }
            for r in results
        ])
        
        return df
    
    def print_cost_summary(self, df: pd.DataFrame):
        """Print formatted cost summary"""
        # Total cost
        total_cost = df["Total Cost"].sum()
        total_time = df["Time (min)"].sum()
        
        self.console.print(Panel(f"[bold green]Total Estimated Cost: ${total_cost:.2f}[/bold green]\n"
                               f"[bold blue]Total Estimated Time: {total_time:.1f} minutes ({total_time/60:.1f} hours)[/bold blue]",
                               title="Cost Summary"))
        
        # Cost by model
        model_costs = df.groupby("Model")["Total Cost"].sum().sort_values(ascending=False)
        model_table = Table(title="Cost by Model")
        model_table.add_column("Model", style="cyan")
        model_table.add_column("Total Cost", style="green")
        model_table.add_column("Percentage", style="yellow")
        
        for model, cost in model_costs.items():
            percentage = (cost / total_cost) * 100
            model_table.add_row(model, f"${cost:.2f}", f"{percentage:.1f}%")
        
        self.console.print(model_table)
        
        # Cost by attack
        attack_costs = df.groupby("Attack")["Total Cost"].sum().sort_values(ascending=False)
        attack_table = Table(title="Cost by Attack")
        attack_table.add_column("Attack", style="cyan")
        attack_table.add_column("Total Cost", style="green")
        attack_table.add_column("Avg Cost per Run", style="yellow")
        
        for attack, cost in attack_costs.items():
            runs = len(df[df["Attack"] == attack])
            avg_cost = cost / runs if runs > 0 else 0
            attack_table.add_row(attack, f"${cost:.2f}", f"${avg_cost:.3f}")
        
        self.console.print(attack_table)
    
    def print_optimization_suggestions(self, df: pd.DataFrame):
        """Print cost optimization suggestions"""
        suggestions = []
        
        # Find most expensive models
        model_costs = df.groupby("Model")["Total Cost"].sum().sort_values(ascending=False)
        expensive_models = model_costs.head(3)
        
        if len(expensive_models) > 0:
            suggestions.append(f"Consider using cheaper alternatives to {', '.join(expensive_models.index[:2])}")
        
        # Find most expensive attacks
        attack_costs = df.groupby("Attack")["Total Cost"].mean().sort_values(ascending=False)
        expensive_attacks = attack_costs.head(3)
        
        if len(expensive_attacks) > 0:
            suggestions.append(f"Multi-turn attacks ({', '.join(expensive_attacks.index[:2])}) are most expensive")
        
        # Suggest batching
        suggestions.append("Use batch API where available for 50% discount")
        suggestions.append("Use prompt caching for repetitive queries (up to 75% discount)")
        suggestions.append("Consider using smaller sample sizes for initial testing")
        
        # Suggest model alternatives
        suggestions.append("Consider DeepSeek models for budget-friendly alternatives")
        suggestions.append("Use mini/nano variants for less critical evaluations")
        
        suggestions_text = "\n".join([f"• {s}" for s in suggestions])
        self.console.print(Panel(suggestions_text, title="💡 Cost Optimization Suggestions"))
    
    def save_detailed_report(self, df: pd.DataFrame, output_file: str):
        """Save detailed cost report to file"""
        # Create summary statistics
        summary_stats = {
            "total_cost": df["Total Cost"].sum(),
            "total_time_minutes": df["Time (min)"].sum(),
            "average_cost_per_experiment": df["Total Cost"].mean(),
            "cost_by_model": df.groupby("Model")["Total Cost"].sum().to_dict(),
            "cost_by_attack": df.groupby("Attack")["Total Cost"].sum().to_dict(),
            "cost_by_defense": df.groupby("Defense")["Total Cost"].sum().to_dict(),
            "cost_by_dataset": df.groupby("Dataset")["Total Cost"].sum().to_dict(),
            "most_expensive_combinations": df.nlargest(10, "Total Cost")[["Model", "Attack", "Defense", "Total Cost"]].to_dict("records"),
            "cheapest_combinations": df.nsmallest(10, "Total Cost")[["Model", "Attack", "Defense", "Total Cost"]].to_dict("records")
        }
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump({
                "summary": summary_stats,
                "detailed_breakdown": df.to_dict("records")
            }, f, indent=2)
        
        self.console.print(f"[green]Detailed report saved to {output_file}[/green]")

def main():
    parser = argparse.ArgumentParser(description="Cost analysis for PromptSecurity experiments")
    parser.add_argument("--config", type=str, help="Path to batch configuration file")
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini", "claude-3-5-sonnet-latest", "gemini-1.5-flash"], 
                       help="Models to analyze")
    parser.add_argument("--datasets", nargs="+", default=["harmbench", "jbb", "airbench"], 
                       help="Datasets to analyze")
    parser.add_argument("--attacks", nargs="+", default=["None", "ReNeLLM", "TapAttack", "ArtPrompt"], 
                       help="Attacks to analyze")
    parser.add_argument("--defenses", nargs="+", default=["None", "JailGuard", "PerplexityFilter"], 
                       help="Defenses to analyze")
    parser.add_argument("--prompts", type=int, default=100, help="Number of prompts per experiment")
    parser.add_argument("--output", type=str, default="cost_analysis_report.json", help="Output file for detailed report")
    
    args = parser.parse_args()
    
    analyzer = CostAnalyzer()
    
    if args.config:
        # Analyze specific batch configuration
        analyzer.console.print(f"[cyan]Analyzing batch configuration: {args.config}[/cyan]")
        results = analyzer.analyze_batch_config(args.config)
        df = pd.DataFrame([
            {
                "Model": r.model_name,
                "Dataset": r.dataset_name,
                "Attack": r.attack_name,
                "Defense": r.defense_name,
                "Prompts": r.prompt_count,
                "Base Cost": r.base_cost,
                "Attack Cost": r.attack_cost,
                "Defense Cost": r.defense_cost,
                "Judger Cost": r.judger_cost,
                "Total Cost": r.total_cost,
                "Time (min)": r.estimated_time_minutes
            }
            for r in results
        ])
    else:
        # Generate comprehensive report
        analyzer.console.print("[cyan]Generating comprehensive cost analysis...[/cyan]")
        df = analyzer.generate_cost_report(
            models=args.models,
            datasets=args.datasets,
            attacks=args.attacks,
            defenses=args.defenses,
            prompt_count=args.prompts
        )
    
    # Print results
    analyzer.print_cost_summary(df)
    analyzer.print_optimization_suggestions(df)
    
    # Save detailed report
    analyzer.save_detailed_report(df, args.output)
    
    # Print local model cost estimates
    analyzer.console.print(Panel(
        "[bold]Local Model Cost Estimates (per hour):[/bold]\n"
        "• RTX 4090: $0.50/hour\n"
        "• A100: $2.00/hour\n"
        "• H100: $4.00/hour\n"
        "• A6000: $1.50/hour\n"
        "• V100: $1.00/hour\n\n"
        "[italic]Note: Local model costs depend on your infrastructure setup[/italic]",
        title="🖥️ Local Model Costs"
    ))

if __name__ == "__main__":
    main()