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