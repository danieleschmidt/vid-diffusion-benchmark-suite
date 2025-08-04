"""Structured logging configuration."""

import os
import sys
import json
import logging
import logging.config
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
            
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        # Add stack trace if present
        if record.stack_info:
            log_entry["stack_trace"] = record.stack_info
            
        return json.dumps(log_entry, ensure_ascii=False)


class StructuredLogger:
    """Wrapper for structured logging with context."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._context = {}
        
    def with_context(self, **kwargs) -> 'StructuredLogger':
        """Create a new logger with additional context."""
        new_logger = StructuredLogger(self.logger)
        new_logger._context = {**self._context, **kwargs}
        return new_logger
        
    def _log(self, level: int, message: str, **kwargs):
        """Log with context and extra fields."""
        extra = {**self._context, **kwargs}
        self.logger.log(level, message, extra={"extra": extra})
        
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)
        
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)
        
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)
        
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)
        
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        extra = {**self._context, **kwargs}
        self.logger.exception(message, extra={"extra": extra})


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_dir: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True
) -> None:
    """Setup comprehensive logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ('json' or 'text')
        log_dir: Directory for log files
        enable_console: Enable console logging
        enable_file: Enable file logging
    """
    log_level = getattr(logging, level.upper())
    
    # Create log directory if needed
    if log_dir and enable_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure formatters
    if format_type == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    # Configure handlers
    handlers = []
    
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        handlers.append(console_handler)
    
    if enable_file and log_dir:
        # Application log file
        app_handler = logging.FileHandler(
            Path(log_dir) / "vid_bench.log"
        )
        app_handler.setFormatter(formatter)
        app_handler.setLevel(log_level)
        handlers.append(app_handler)
        
        # Error log file
        error_handler = logging.FileHandler(
            Path(log_dir) / "errors.log"
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        handlers.append(error_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )
    
    # Configure third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    # Set SQLAlchemy logging
    if level.upper() == "DEBUG":
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
        logging.getLogger("sqlalchemy.pool").setLevel(logging.INFO)
    else:
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)


def get_structured_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(logging.getLogger(name))


def configure_logging_from_env():
    """Configure logging from environment variables."""
    level = os.getenv("VID_BENCH_LOG_LEVEL", "INFO")
    format_type = os.getenv("VID_BENCH_LOG_FORMAT", "json")
    log_dir = os.getenv("VID_BENCH_LOG_DIR")
    
    enable_console = os.getenv("VID_BENCH_LOG_CONSOLE", "true").lower() == "true"
    enable_file = os.getenv("VID_BENCH_LOG_FILE", "true").lower() == "true"
    
    setup_logging(
        level=level,
        format_type=format_type,
        log_dir=log_dir,
        enable_console=enable_console,
        enable_file=enable_file
    )


# Configure logging on import if not already configured
if not logging.getLogger().handlers:
    configure_logging_from_env()


class BenchmarkContextLogger(StructuredLogger):
    """Logger with benchmark-specific context."""
    
    def __init__(self, logger: logging.Logger, model_name: str = None, run_id: str = None):
        super().__init__(logger)
        context = {}
        if model_name:
            context["model_name"] = model_name
        if run_id:
            context["run_id"] = run_id
        self._context = context
        
    def log_benchmark_start(self, prompts_count: int, config: Dict[str, Any]):
        """Log benchmark start."""
        self.info(
            "Benchmark started",
            prompts_count=prompts_count,
            config=config,
            event="benchmark_start"
        )
        
    def log_benchmark_complete(self, success_rate: float, duration: float):
        """Log benchmark completion."""
        self.info(
            "Benchmark completed",
            success_rate=success_rate,
            duration_seconds=duration,
            event="benchmark_complete"
        )
        
    def log_prompt_result(self, prompt_idx: int, prompt: str, status: str, duration: float):
        """Log individual prompt result."""
        self.info(
            "Prompt processing completed",
            prompt_index=prompt_idx,
            prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt,
            status=status,
            duration_seconds=duration,
            event="prompt_complete"
        )
        
    def log_metrics_computed(self, metrics: Dict[str, float]):
        """Log metrics computation."""
        self.info(
            "Quality metrics computed",
            **metrics,
            event="metrics_computed"
        )
        
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context."""
        extra_context = context or {}
        self.error(
            f"Error occurred: {str(error)}",
            error_type=type(error).__name__,
            **extra_context,
            event="error"
        )


def get_benchmark_logger(name: str, model_name: str = None, run_id: str = None) -> BenchmarkContextLogger:
    """Get a benchmark-specific logger.
    
    Args:
        name: Logger name
        model_name: Model being benchmarked
        run_id: Evaluation run ID
        
    Returns:
        BenchmarkContextLogger instance
    """
    return BenchmarkContextLogger(logging.getLogger(name), model_name, run_id)