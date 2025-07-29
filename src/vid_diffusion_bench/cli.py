"""Command-line interface for vid-bench."""

import click
from pathlib import Path
from .benchmark import BenchmarkSuite
from .prompts import StandardPrompts


@click.group()
@click.version_option()
def main():
    """Video Diffusion Benchmark Suite CLI."""
    pass


@main.command()
@click.option("--model", required=True, help="Model name to evaluate")
@click.option("--prompts", default="diverse", help="Prompt set to use")
@click.option("--output", type=click.Path(), help="Output results file")
def evaluate(model: str, prompts: str, output: str):
    """Evaluate a single model."""
    suite = BenchmarkSuite()
    
    # Select prompt set
    if prompts == "diverse":
        prompt_list = StandardPrompts.DIVERSE_SET_V2
    else:
        prompt_list = [prompts]  # Single prompt
        
    results = suite.evaluate_model(model, prompt_list)
    
    click.echo(f"Model: {model}")
    click.echo(f"FVD: {results['fvd']:.2f}")
    click.echo(f"Latency: {results['latency']:.2f}s")


@main.command()
@click.option("--models", default="all", help="Models to benchmark")
@click.option("--output", required=True, type=click.Path(), help="Output file")
def benchmark(models: str, output: str):
    """Run full benchmark suite."""
    click.echo(f"Running benchmark for models: {models}")
    click.echo(f"Results will be saved to: {output}")


if __name__ == "__main__":
    main()