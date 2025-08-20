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
@click.option("--models", default="all", help="Models to benchmark (comma-separated or 'all')")
@click.option("--output", required=True, type=click.Path(), help="Output file")
@click.option("--prompts", default="standard", help="Prompt set: standard, diverse, custom")
@click.option("--parallel", default=2, help="Number of parallel evaluations")
@click.option("--timeout", default=300, help="Timeout per model in seconds")
def benchmark(models: str, output: str, prompts: str, parallel: int, timeout: int):
    """Run full benchmark suite."""
    from .models.registry import list_models
    from .prompts import StandardPrompts
    import json
    from datetime import datetime
    
    # Parse model list
    if models == "all":
        model_list = list_models()
    else:
        model_list = [m.strip() for m in models.split(",")]
    
    # Select prompt set
    if prompts == "diverse":
        prompt_list = StandardPrompts.DIVERSE_SET_V2
    elif prompts == "standard":
        prompt_list = StandardPrompts.DIVERSE_SET_V2[:10]  # First 10
    else:
        prompt_list = [prompts]
    
    click.echo(f"🚀 Running benchmark for {len(model_list)} models")
    click.echo(f"📝 Using {len(prompt_list)} prompts")
    click.echo(f"⚡ Parallel jobs: {parallel}")
    
    suite = BenchmarkSuite()
    results = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
        "summary": {}
    }
    
    for i, model_name in enumerate(model_list, 1):
        click.echo(f"\n[{i}/{len(model_list)}] Evaluating {model_name}...")
        try:
            model_results = suite.evaluate_model(model_name, prompt_list, timeout=timeout)
            results["models"][model_name] = model_results
            click.echo(f"✅ {model_name}: FVD={model_results.get('fvd', 'N/A'):.2f}")
        except Exception as e:
            click.echo(f"❌ {model_name}: {str(e)}")
            results["models"][model_name] = {"error": str(e)}
    
    # Save results
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    click.echo(f"\n📊 Results saved to: {output}")


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
def list_models():
    """List all available models."""
    from .models.registry import list_models
    
    models = list_models()
    if not models:
        click.echo("No models registered.")
        return
        
    click.echo("Available models:")
    for model in sorted(models):
        try:
            from .models.registry import get_model
            model_instance = get_model(model)
            reqs = model_instance.requirements
            
            click.echo(f"\n{model}:")
            click.echo(f"  VRAM: {reqs.get('vram_gb', 'N/A')} GB")
            click.echo(f"  Precision: {reqs.get('precision', 'N/A')}")
            click.echo(f"  Model Size: {reqs.get('model_size_gb', 'N/A')} GB")
            click.echo(f"  Max Frames: {reqs.get('max_frames', 'N/A')}")
        except Exception as e:
            click.echo(f"  - {model} (failed to load - {e})")


@main.command()
@click.argument("model_name")
@click.option("--prompt", default="A cat playing piano", help="Test prompt")
@click.option("--frames", default=16, help="Number of frames")
@click.option("--fps", default=8, help="Frames per second")
def test_model(model_name: str, prompt: str, frames: int, fps: int):
    """Test a single model with a prompt."""
    from .models.registry import get_model
    from .exceptions import ModelNotFoundError
    
    try:
        model = get_model(model_name)
        click.echo(f"Testing model: {model_name}")
        click.echo(f"Prompt: {prompt}")
        
        import time
        start_time = time.time()
        video = model.generate(prompt, num_frames=frames, fps=fps)
        duration = time.time() - start_time
        
        click.echo(f"✅ Generation successful!")
        click.echo(f"Video shape: {video.shape}")
        click.echo(f"Duration: {duration:.2f}s")
        
    except ModelNotFoundError as e:
        click.echo(f"❌ {e.message}")
        if e.available_models:
            click.echo("Available models:")
            for model in e.available_models:
                click.echo(f"  - {model}")
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")


@main.command()
@click.option("--models", default="all", help="Models to compare (comma-separated)")
@click.option("--prompts", default="standard", help="Prompt set for comparison")
@click.option("--output", type=click.Path(), help="Output comparison report")
@click.option("--metrics", default="fvd,latency,vram", help="Metrics to compare")
def compare(models: str, prompts: str, output: str, metrics: str):
    """Compare multiple models side-by-side."""
    from .models.registry import list_models
    from .prompts import StandardPrompts
    import json
    from datetime import datetime
    
    # Parse models
    if models == "all":
        model_list = list_models()[:5]  # Limit to 5 for comparison
    else:
        model_list = [m.strip() for m in models.split(",")]
    
    if len(model_list) < 2:
        click.echo("❌ Need at least 2 models for comparison")
        return
    
    # Select prompts
    if prompts == "standard":
        prompt_list = StandardPrompts.DIVERSE_SET_V2[:5]
    else:
        prompt_list = [prompts]
    
    click.echo(f"🔄 Comparing {len(model_list)} models on {len(prompt_list)} prompts")
    
    suite = BenchmarkSuite()
    comparison_results = {
        "timestamp": datetime.now().isoformat(),
        "models": model_list,
        "prompts": prompt_list,
        "results": {},
        "ranking": {}
    }
    
    # Evaluate each model
    for model_name in model_list:
        click.echo(f"Evaluating {model_name}...")
        try:
            results = suite.evaluate_model(model_name, prompt_list)
            comparison_results["results"][model_name] = results
        except Exception as e:
            click.echo(f"❌ {model_name}: {str(e)}")
    
    # Display comparison table
    click.echo("\n📊 COMPARISON RESULTS")
    click.echo("=" * 80)
    click.echo(f"{'Model':<20} {'FVD':<10} {'Latency(s)':<12} {'VRAM(GB)':<10} {'Score':<8}")
    click.echo("-" * 80)
    
    for model_name in model_list:
        if model_name in comparison_results["results"]:
            r = comparison_results["results"][model_name]
            fvd = r.get("fvd", "N/A")
            latency = r.get("latency", "N/A")
            vram = r.get("vram_gb", "N/A")
            score = r.get("overall_score", "N/A")
            click.echo(f"{model_name:<20} {fvd:<10} {latency:<12} {vram:<10} {score:<8}")
    
    # Save detailed results
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        click.echo(f"\n📄 Detailed results saved to: {output}")


@main.command()
@click.option("--experiment", required=True, help="Research experiment name")
@click.option("--hypothesis", required=True, help="Research hypothesis to test")
@click.option("--models", default="all", help="Models for research study")
@click.option("--dataset", default="standard", help="Dataset/prompts for evaluation")
@click.option("--runs", default=3, help="Number of runs for statistical significance")
@click.option("--output-dir", type=click.Path(), help="Output directory for research results")
def research(experiment: str, hypothesis: str, models: str, dataset: str, runs: int, output_dir: str):
    """Run research-grade benchmarking with statistical analysis."""
    from .research_framework import ResearchFramework
    from .models.registry import list_models
    from datetime import datetime
    
    click.echo(f"🔬 Starting research experiment: {experiment}")
    click.echo(f"💡 Hypothesis: {hypothesis}")
    
    # Setup research framework
    framework = ResearchFramework(
        experiment_name=experiment,
        hypothesis=hypothesis,
        output_dir=output_dir or f"research/{experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Parse models
    if models == "all":
        model_list = list_models()
    else:
        model_list = [m.strip() for m in models.split(",")]
    
    click.echo(f"📋 Models: {', '.join(model_list)}")
    click.echo(f"🔄 Runs per model: {runs}")
    
    # Run research study
    try:
        results = framework.run_experiment(
            models=model_list,
            dataset=dataset,
            num_runs=runs
        )
        
        click.echo(f"\n📊 RESEARCH RESULTS")
        click.echo("=" * 60)
        
        if results.get("statistical_significance"):
            click.echo("✅ Statistically significant results found!")
        else:
            click.echo("⚠️  Results not statistically significant")
        
        click.echo(f"📈 Best performing model: {results.get('best_model', 'Unknown')}")
        click.echo(f"📉 P-value: {results.get('p_value', 'N/A')}")
        
        # Generate publication-ready report
        report_path = framework.generate_publication_report()
        click.echo(f"📝 Publication report: {report_path}")
        
    except Exception as e:
        click.echo(f"❌ Research experiment failed: {str(e)}")


@main.command()
@click.option("--db-url", help="Database URL (defaults to local SQLite)")
def init_db(db_url: str):
    """Initialize the benchmark database."""
    from .database.connection import init_database
    
    click.echo("🗄️  Initializing benchmark database...")
    
    try:
        init_database(db_url)
        click.echo("✅ Database initialized successfully!")
    except Exception as e:
        click.echo(f"❌ Database initialization failed: {str(e)}")


@main.command()
@click.option("--output", type=click.Path(), help="Output file for detailed report")
def health_check(output: str):
    """Run system health checks."""
    import torch
    import psutil
    import json
    from datetime import datetime
    from .models.registry import list_models
    
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "python_version": __import__("sys").version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        },
        "models": {
            "registered_count": len(list_models()),
            "available_models": list(list_models())
        },
        "status": "healthy"
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            health_status["system"][f"gpu_{i}"] = {
                "name": gpu_props.name,
                "memory_gb": round(gpu_props.total_memory / (1024**3), 2),
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}"
            }
    
    if output:
        with open(output, 'w') as f:
            json.dump(health_status, f, indent=2)
        click.echo(f"Health report saved to {output}")
    else:
        click.echo("🏥 System Health Check")
        click.echo("=" * 30)
        click.echo(f"Status: {health_status['status']}")
        click.echo(f"CUDA Available: {health_status['system']['cuda_available']}")
        click.echo(f"GPU Devices: {health_status['system']['cuda_devices']}")
        click.echo(f"Registered Models: {health_status['models']['registered_count']}")
        click.echo(f"Available Memory: {health_status['system']['memory_gb']}GB")


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
            .model-name {{ color: #2c3e50; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Video Diffusion Model Comparison Report</h1>
        <p>Generated on: {comparison.get('timestamp', 'N/A')}</p>
        
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
        if hasattr(result, 'metrics') and result.metrics and hasattr(result, 'performance') and result.performance:
            html_content += f"""
            <tr>
                <td class="model-name">{name}</td>
                <td>{result.metrics.get('overall_score', 0):.1f}</td>
                <td>{result.metrics.get('fvd', 0):.1f}</td>
                <td>{result.metrics.get('clip_similarity', 0):.3f}</td>
                <td>{result.performance.get('avg_latency_ms', 0):.1f}</td>
                <td>{result.performance.get('peak_vram_gb', 0):.1f}</td>
                <td>{getattr(result, 'success_rate', 0):.1%}</td>
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
            if hasattr(result, 'metrics') and result.metrics and hasattr(result, 'performance') and result.performance:
                writer.writerow([
                    name,
                    f"{result.metrics.get('overall_score', 0):.1f}",
                    f"{result.metrics.get('fvd', 0):.1f}",
                    f"{result.metrics.get('clip_similarity', 0):.3f}",
                    f"{result.performance.get('avg_latency_ms', 0):.1f}",
                    f"{result.performance.get('peak_vram_gb', 0):.1f}",
                    f"{getattr(result, 'success_rate', 0):.1%}"
                ])


@main.command()
@click.option("--output-dir", default="./research", help="Output directory for research data")
@click.option("--models", multiple=True, help="Models to include in research study")
@click.option("--seeds", default=3, type=int, help="Number of random seeds for reproducibility")
@click.option("--samples-per-seed", default=5, type=int, help="Samples per seed")
def research(output_dir: str, models: tuple, seeds: int, samples_per_seed: int):
    """Run research-grade benchmarks with statistical analysis."""
    from .research_framework import run_research_benchmark
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
        
    except Exception as e:
        click.echo(f"Research benchmark failed: {e}", err=True)


if __name__ == "__main__":
    main()