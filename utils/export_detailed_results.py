#!/usr/bin/env python3
"""
Export Detailed Attack Results
==============================

This script extracts and exports all detailed attack prompts and responses
from the evaluation checkpoint files.
"""

import json
import csv
import pandas as pd
from pathlib import Path
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def load_checkpoint(checkpoint_path):
    """Load checkpoint data from JSON file"""
    try:
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading checkpoint: {e}[/red]")
        return None

def extract_detailed_results(checkpoint_data):
    """Extract all individual results from checkpoint data"""
    
    detailed_results = []
    
    for combination in checkpoint_data.get('results', []):
        attack_name = combination.get('attack_name', 'Unknown')
        defense_name = combination.get('defense_name', 'Unknown')
        
        # Extract individual results
        individual_results = combination.get('individual_results', [])
        
        for result in individual_results:
            detailed_result = {
                'attack_name': attack_name,
                'defense_name': defense_name,
                'prompt_index': result.get('prompt_index', 0),
                'original_prompt': result.get('original_prompt', ''),
                'crafted_prompt': result.get('crafted_prompt', ''),
                'response': result.get('response', ''),
                'judgment_score': result.get('judgment_score', 'N/A'),
                'is_successful': result.get('is_successful', False),
                'queries_used': result.get('queries_used', 0),
                'execution_time': result.get('execution_time', 0.0),
                'error': result.get('error', '')
            }
            detailed_results.append(detailed_result)
    
    return detailed_results

def export_to_csv(detailed_results, output_path):
    """Export detailed results to CSV file"""
    
    try:
        df = pd.DataFrame(detailed_results)
        df.to_csv(output_path, index=False, encoding='utf-8')
        console.print(f"[green]✅ Exported {len(detailed_results)} detailed results to {output_path}[/green]")
        return True
    except Exception as e:
        console.print(f"[red]❌ Error exporting to CSV: {e}[/red]")
        return False

def export_successful_attacks(detailed_results, output_path):
    """Export only successful attacks to a separate file"""
    
    successful_attacks = [r for r in detailed_results if r['is_successful'] == True]
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Successful Attack Prompts and Responses\n\n")
            
            for i, result in enumerate(successful_attacks, 1):
                f.write(f"## Attack #{i}\n")
                f.write(f"**Attack**: {result['attack_name']}\n")
                f.write(f"**Defense**: {result['defense_name']}\n")
                f.write(f"**Original Prompt**: {result['original_prompt']}\n\n")
                f.write(f"**Crafted Prompt**:\n```\n{result['crafted_prompt']}\n```\n\n")
                f.write(f"**Model Response**:\n```\n{result['response']}\n```\n\n")
                f.write(f"**Judgment Score**: {result['judgment_score']}\n")
                f.write(f"**Queries Used**: {result['queries_used']}\n")
                f.write(f"**Execution Time**: {result['execution_time']:.2f}s\n")
                f.write("\n" + "="*80 + "\n\n")
        
        console.print(f"[green]✅ Exported {len(successful_attacks)} successful attacks to {output_path}[/green]")
        return True
    except Exception as e:
        console.print(f"[red]❌ Error exporting successful attacks: {e}[/red]")
        return False

def print_summary(detailed_results):
    """Print summary statistics"""
    
    if not detailed_results:
        console.print("[yellow]No detailed results found[/yellow]")
        return
    
    total_results = len(detailed_results)
    successful_attacks = sum(1 for r in detailed_results if r['is_successful'] == True)
    unique_attacks = len(set(r['attack_name'] for r in detailed_results))
    unique_defenses = len(set(r['defense_name'] for r in detailed_results))
    
    # Create summary table
    table = Table(title="Detailed Results Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Test Cases", str(total_results))
    table.add_row("Successful Attacks", str(successful_attacks))
    table.add_row("Success Rate", f"{(successful_attacks/total_results*100):.1f}%" if total_results > 0 else "0%")
    table.add_row("Unique Attacks", str(unique_attacks))
    table.add_row("Unique Defenses", str(unique_defenses))
    
    console.print(table)
    
    # Show top performing attacks
    if successful_attacks > 0:
        attack_success = {}
        for r in detailed_results:
            attack = r['attack_name']
            if attack not in attack_success:
                attack_success[attack] = {'total': 0, 'successful': 0}
            attack_success[attack]['total'] += 1
            if r['is_successful']:
                attack_success[attack]['successful'] += 1
        
        console.print("\n[bold cyan]Top Performing Attacks:[/bold cyan]")
        for attack, stats in sorted(attack_success.items(), 
                                   key=lambda x: x[1]['successful']/x[1]['total'] if x[1]['total'] > 0 else 0, 
                                   reverse=True)[:5]:
            success_rate = (stats['successful']/stats['total']*100) if stats['total'] > 0 else 0
            console.print(f"  {attack}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Export detailed attack results")
    parser.add_argument("--checkpoint", 
                       default="experiments/results/harmbench_full/experiment_checkpoint.json",
                       help="Path to checkpoint file")
    parser.add_argument("--output-dir", 
                       default="experiments/results/harmbench_full/exported_details",
                       help="Output directory for exported files")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(Panel(
        f"[bold cyan]📊 Exporting Detailed Attack Results[/bold cyan]\n"
        f"Checkpoint: [bold]{args.checkpoint}[/bold]\n"
        f"Output Directory: [bold]{output_dir}[/bold]",
        title="Attack Results Exporter"
    ))
    
    # Load checkpoint
    checkpoint_data = load_checkpoint(args.checkpoint)
    if not checkpoint_data:
        return 1
    
    # Extract detailed results
    detailed_results = extract_detailed_results(checkpoint_data)
    
    # Print summary
    print_summary(detailed_results)
    
    if not detailed_results:
        console.print("[yellow]No detailed results to export[/yellow]")
        return 0
    
    # Export to CSV
    csv_path = output_dir / "detailed_results.csv"
    export_to_csv(detailed_results, csv_path)
    
    # Export successful attacks
    successful_path = output_dir / "successful_attacks.md"
    export_successful_attacks(detailed_results, successful_path)
    
    # Export full data as JSON
    json_path = output_dir / "detailed_results.json"
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        console.print(f"[green]✅ Exported full data to {json_path}[/green]")
    except Exception as e:
        console.print(f"[red]❌ Error exporting JSON: {e}[/red]")
    
    console.print(f"\n[bold green]🎉 Export completed! Files saved to {output_dir}[/bold green]")
    return 0

if __name__ == "__main__":
    exit(main())