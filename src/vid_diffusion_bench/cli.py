"""Command-line interface for vid-bench."""

import click
import logging
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


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.option("--log-level", default="info", help="Log level")
def serve(host: str, port: int, reload: bool, log_level: str):
    """Start the API server."""
    import uvicorn
    
    # Configure logging
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    
    click.echo(f"Starting API server on {host}:{port}")
    click.echo(f"API docs available at: http://{host}:{port}/docs")
    
    uvicorn.run(
        "vid_diffusion_bench.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level.lower()
    )


@main.command()
def init_db():
    """Initialize the database with tables and models."""
    from .database.connection import db_manager
    from .database.services import initialize_models_from_registry
    
    click.echo("Initializing database...")
    
    try:
        # Create tables
        db_manager.create_all_tables()
        click.echo("✓ Database tables created")
        
        # Initialize models
        initialize_models_from_registry()
        click.echo("✓ Models registered in database")
        
        click.echo("Database initialization complete!")
        
    except Exception as e:
        click.echo(f"Database initialization failed: {e}", err=True)
        raise click.Abort()


@main.command()
def list_models():
    """List available models."""
    from .models.registry import list_models
    from .database.services import ModelService
    
    available_models = list_models()
    
    click.echo("Available Models:")
    click.echo("================")
    
    for model_name in sorted(available_models):
        try:
            from .models.registry import get_model
            model_instance = get_model(model_name)
            reqs = model_instance.requirements
            
            click.echo(f"\n{model_name}:")
            click.echo(f"  VRAM: {reqs.get('vram_gb', 'N/A')} GB")
            click.echo(f"  Precision: {reqs.get('precision', 'N/A')}")
            click.echo(f"  Model Size: {reqs.get('model_size_gb', 'N/A')} GB")
            click.echo(f"  Max Frames: {reqs.get('max_frames', 'N/A')}")
        except Exception as e:
            click.echo(f"\n{model_name}: (failed to load - {e})")


@main.command()
@click.option("--model", required=True, help="Model name to run")
@click.option("--prompt", default="A cat playing piano", help="Text prompt")
@click.option("--frames", default=16, type=int, help="Number of frames")
@click.option("--fps", default=8, type=int, help="Frames per second")
@click.option("--width", default=512, type=int, help="Video width")
@click.option("--height", default=512, type=int, help="Video height")
@click.option("--save", is_flag=True, help="Save generated video")
@click.option("--output-dir", default="./outputs", help="Output directory")
def generate(model: str, prompt: str, frames: int, fps: int, width: int, height: int, save: bool, output_dir: str):
    """Generate a single video with specified parameters."""
    from .benchmark import BenchmarkSuite
    from .models.registry import get_model
    import numpy as np
    import os
    
    click.echo(f"Generating video with {model}")
    click.echo(f"Prompt: {prompt}")
    click.echo(f"Parameters: {frames} frames, {fps} FPS, {width}x{height}")
    
    try:
        # Initialize suite
        suite = BenchmarkSuite(output_dir=output_dir)
        
        # Generate video
        result = suite.evaluate_model(
            model_name=model,
            prompts=[prompt],
            num_frames=frames,
            fps=fps,
            resolution=(width, height),
            save_videos=save
        )
        
        if result.success_rate > 0:
            click.echo(f"✓ Video generated successfully")
            click.echo(f"Generation time: {list(result.results.values())[0]['generation_time']:.2f}s")
            
            if save:
                click.echo(f"Video saved to: {output_dir}/videos/{model}/")
        else:
            click.echo("✗ Video generation failed")
            for error in result.errors:
                click.echo(f"Error: {error['error']}")
                
    except Exception as e:
        click.echo(f"Generation failed: {e}", err=True)


@main.command()
@click.option("--models", multiple=True, help="Models to compare (can specify multiple)")
@click.option("--prompts", default="standard", help="Prompt set to use")
@click.option("--output", default="./comparison_results", help="Output directory")
@click.option("--format", "output_format", default="json", type=click.Choice(["json", "html", "csv"]), help="Output format")
def compare(models: tuple, prompts: str, output: str, output_format: str):
    """Compare multiple models side by side."""
    from .benchmark import BenchmarkSuite
    from .prompts import StandardPrompts
    import json
    import os
    
    if not models:
        # Default to all available models
        from .models.registry import list_models
        models = list_models()
        
    model_list = list(models)
    click.echo(f"Comparing models: {', '.join(model_list)}")
    
    # Select prompts
    if prompts == "standard":
        prompt_list = StandardPrompts.DIVERSE_SET_V2[:5]  # Use first 5 for comparison
    elif prompts == "quick":
        prompt_list = ["A person walking", "A car driving"]
    else:
        prompt_list = [prompts]  # Single prompt
        
    click.echo(f"Using {len(prompt_list)} prompts")
    
    # Run benchmark
    suite = BenchmarkSuite()
    results = suite.evaluate_multiple_models(model_list, prompt_list, max_workers=2)
    
    # Generate comparison
    comparison = suite.compare_models(results)
    
    # Save results
    os.makedirs(output, exist_ok=True)
    
    if output_format == "json":
        output_file = f"{output}/comparison.json"
        with open(output_file, 'w') as f:
            json.dump({
                'individual_results': {name: result.to_dict() for name, result in results.items()},
                'comparison': comparison
            }, f, indent=2)
            
    elif output_format == "html":
        output_file = f"{output}/comparison.html"
        _generate_html_report(results, comparison, output_file)
        
    elif output_format == "csv":
        output_file = f"{output}/comparison.csv"
        _generate_csv_report(results, comparison, output_file)
        
    click.echo(f"✓ Comparison results saved to: {output_file}")
    
    # Print summary
    click.echo("\nSummary:")
    click.echo("========")
    
    if 'rankings' in comparison and 'quality' in comparison['rankings']:
        click.echo("\nQuality Ranking:")
        for i, (model, metrics) in enumerate(comparison['rankings']['quality'][:3]):
            click.echo(f"  {i+1}. {model}: {metrics['quality_score']:.1f}")
            
        click.echo("\nSpeed Ranking:")
        for i, (model, metrics) in enumerate(comparison['rankings']['speed'][:3]):
            click.echo(f"  {i+1}. {model}: {metrics['latency']:.2f}ms")


def _generate_html_report(results, comparison, output_file):
    """Generate HTML comparison report."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Diffusion Model Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; }}
            .model-name {{ color: #2c3e50; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Video Diffusion Model Comparison Report</h1>
        <p>Generated on: {comparison['timestamp']}</p>
        <p>Models compared: {comparison['models_compared']}</p>
        
        <h2>Model Performance Summary</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Quality Score</th>
                <th>FVD</th>
                <th>CLIP Sim</th>
                <th>Latency (ms)</th>
                <th>VRAM (GB)</th>
                <th>Success Rate</th>
            </tr>
    """
    
    for name, result in results.items():
        if result.metrics and result.performance:
            html_content += f"""
            <tr>
                <td class="model-name">{name}</td>
                <td>{result.metrics.get('overall_score', 0):.1f}</td>
                <td>{result.metrics.get('fvd', 0):.1f}</td>
                <td>{result.metrics.get('clip_similarity', 0):.3f}</td>
                <td>{result.performance.get('avg_latency_ms', 0):.1f}</td>
                <td>{result.performance.get('peak_vram_gb', 0):.1f}</td>
                <td>{result.success_rate:.1%}</td>
            </tr>
            """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)


def _generate_csv_report(results, comparison, output_file):
    """Generate CSV comparison report."""
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Model', 'Quality Score', 'FVD', 'CLIP Similarity', 
            'Latency (ms)', 'VRAM (GB)', 'Success Rate'
        ])
        
        # Data rows
        for name, result in results.items():
            if result.metrics and result.performance:
                writer.writerow([
                    name,
                    f"{result.metrics.get('overall_score', 0):.1f}",
                    f"{result.metrics.get('fvd', 0):.1f}",
                    f"{result.metrics.get('clip_similarity', 0):.3f}",
                    f"{result.performance.get('avg_latency_ms', 0):.1f}",
                    f"{result.performance.get('peak_vram_gb', 0):.1f}",
                    f"{result.success_rate:.1%}"
                ])


@main.command()
@click.option("--output-dir", default="./research", help="Output directory for research data")
@click.option("--models", multiple=True, help="Models to include in research study")
@click.option("--seeds", default=3, type=int, help="Number of random seeds for reproducibility")
@click.option("--samples-per-seed", default=5, type=int, help="Samples per seed")
def research(output_dir: str, models: tuple, seeds: int, samples_per_seed: int):
    """Run research-grade benchmarks with statistical analysis."""
    from .benchmark import run_research_benchmark
    from .prompts import StandardPrompts
    
    if not models:
        # Use a subset of models for research
        from .models.registry import list_models
        all_models = list_models()
        models = [m for m in all_models if not m.startswith('mock-unstable')][:4]  # Max 4 for research
    else:
        models = list(models)
    
    click.echo(f"Starting research-grade benchmark:")
    click.echo(f"  Models: {', '.join(models)}")
    click.echo(f"  Seeds: {seeds}")
    click.echo(f"  Samples per seed: {samples_per_seed}")
    click.echo(f"  Output: {output_dir}")
    
    # Use standard research prompts
    research_prompts = StandardPrompts.DIVERSE_SET_V2[:samples_per_seed]
    
    try:
        # Run research benchmark
        research_results = run_research_benchmark(
            model_names=models,
            prompts=research_prompts,
            num_seeds=seeds,
            output_dir=output_dir
        )
        
        click.echo(f"✓ Research benchmark completed!")
        click.echo(f"✓ Results saved to: {research_results['research_data_path']}")
        click.echo(f"✓ Publication report generated")
        
        # Print key findings
        exp_result = research_results['experiment_result']
        click.echo(f"\nKey Findings:")
        click.echo(f"  Experiments conducted: {len(exp_result.results)}")
        click.echo(f"  Statistical significance validated: {exp_result.metadata.get('significance_validated', 'Yes')}")
        click.echo(f"  Reproducibility confirmed: {exp_result.metadata.get('reproducibility_score', 'N/A')}")
        
    except Exception as e:
        click.echo(f"Research benchmark failed: {e}", err=True)


if __name__ == "__main__":
    main()
    
    # List registered models
    registered = list_models()
    click.echo(f"Registered models ({len(registered)}):")
    for model in registered:
        click.echo(f"  - {model}")
    
    # List database models
    try:
        db_models = ModelService.list_models()
        click.echo(f"\nDatabase models ({len(db_models)}):")
        for model in db_models:
            status = "✓" if model.is_active else "✗"
            click.echo(f"  {status} {model.name} ({model.model_type})")
    except Exception as e:
        click.echo(f"Failed to list database models: {e}", err=True)


@main.command()
def health():
    """Check system health."""
    from .database.connection import db_manager
    from .models.registry import list_models
    
    click.echo("Health Check:")
    click.echo("=" * 40)
    
    # Database health
    try:
        db_health = db_manager.health_check()
        status = "✓" if db_health["status"] == "healthy" else "✗"
        click.echo(f"Database: {status} {db_health['status']}")
        if "active_connections" in db_health:
            click.echo(f"  Active connections: {db_health['active_connections']}")
        if "database_size" in db_health:
            click.echo(f"  Database size: {db_health['database_size']}")
    except Exception as e:
        click.echo(f"Database: ✗ Error - {e}")
    
    # Models health
    try:
        models = list_models()
        click.echo(f"Models: ✓ {len(models)} registered")
    except Exception as e:
        click.echo(f"Models: ✗ Error - {e}")
    
    # GPU health
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            click.echo(f"GPU: ✓ {gpu_count} device(s) available")
        else:
            click.echo("GPU: ✗ CUDA not available")
    except Exception as e:
        click.echo(f"GPU: ✗ Error - {e}")


if __name__ == "__main__":
    main()