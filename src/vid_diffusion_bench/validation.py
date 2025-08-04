"""Input validation and sanitization utilities."""

import re
import os
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

from .exceptions import InvalidPromptError, InvalidConfigError, ValidationError


class PromptValidator:
    """Validates and sanitizes text prompts."""
    
    # Dangerous patterns to filter
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # JavaScript
        r'javascript:',               # JavaScript URLs
        r'data:.*script',            # Data URLs with scripts
        r'<iframe[^>]*>.*?</iframe>', # Iframes
        r'<object[^>]*>.*?</object>', # Objects
        r'<embed[^>]*>.*?</embed>',   # Embeds
        r'<link[^>]*>',              # Link tags
        r'<meta[^>]*>',              # Meta tags
        r'on\w+\s*=',                # Event handlers
        r'eval\s*\(',                # eval() calls
        r'exec\s*\(',                # exec() calls
    ]
    
    # SQL injection patterns
    SQL_PATTERNS = [
        r'\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b',
        r'[\'";]',  # SQL delimiters
        r'--',      # SQL comments
        r'/\*.*?\*/',  # SQL block comments
    ]
    
    # Path traversal patterns
    PATH_PATTERNS = [
        r'\.\.',     # Parent directory
        r'~/',       # Home directory
        r'/etc/',    # System directories
        r'/var/',
        r'/usr/',
        r'/root/',
        r'[\\\/]',   # Directory separators (be careful with this)
    ]
    
    def __init__(self, max_length: int = 1000, allow_html: bool = False):
        """Initialize prompt validator.
        
        Args:
            max_length: Maximum allowed prompt length
            allow_html: Whether to allow HTML tags
        """
        self.max_length = max_length
        self.allow_html = allow_html
        
    def validate(self, prompt: str) -> str:
        """Validate and sanitize a prompt.
        
        Args:
            prompt: Input prompt to validate
            
        Returns:
            Sanitized prompt
            
        Raises:
            InvalidPromptError: If prompt is invalid
        """
        if not isinstance(prompt, str):
            raise InvalidPromptError(str(prompt), "Prompt must be a string")
            
        # Check length
        if len(prompt) > self.max_length:
            raise InvalidPromptError(
                prompt, 
                f"Prompt too long: {len(prompt)} > {self.max_length} characters"
            )
            
        # Check for empty or whitespace-only prompts
        if not prompt.strip():
            raise InvalidPromptError(prompt, "Prompt cannot be empty")
            
        # Check for dangerous patterns
        if not self.allow_html:
            self._check_dangerous_patterns(prompt)
            
        # Check for SQL injection
        self._check_sql_injection(prompt)
        
        # Check for path traversal
        self._check_path_traversal(prompt)
        
        # Sanitize the prompt
        sanitized = self._sanitize(prompt)
        
        return sanitized
        
    def validate_batch(self, prompts: List[str]) -> List[str]:
        """Validate a batch of prompts.
        
        Args:
            prompts: List of prompts to validate
            
        Returns:
            List of sanitized prompts
        """
        if len(prompts) > 100:
            raise ValidationError("Too many prompts in batch (max 100)")
            
        return [self.validate(prompt) for prompt in prompts]
        
    def _check_dangerous_patterns(self, prompt: str):
        """Check for dangerous HTML/JS patterns."""
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                raise InvalidPromptError(
                    prompt, 
                    f"Prompt contains potentially dangerous pattern: {pattern}"
                )
                
    def _check_sql_injection(self, prompt: str):
        """Check for SQL injection patterns."""
        for pattern in self.SQL_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                raise InvalidPromptError(
                    prompt,
                    f"Prompt contains potential SQL injection pattern: {pattern}"
                )
                
    def _check_path_traversal(self, prompt: str):
        """Check for path traversal patterns."""
        for pattern in self.PATH_PATTERNS:
            if re.search(pattern, prompt):
                raise InvalidPromptError(
                    prompt,
                    f"Prompt contains potential path traversal pattern: {pattern}"
                )
                
    def _sanitize(self, prompt: str) -> str:
        """Sanitize prompt by removing/escaping dangerous content."""
        # Remove null bytes
        sanitized = prompt.replace('\x00', '')
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        # Remove control characters except newlines and tabs
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\n\t')
        
        return sanitized.strip()


class ConfigValidator:
    """Validates benchmark configuration parameters."""
    
    # Valid configuration ranges
    CONFIG_RANGES = {
        "num_frames": (1, 256),
        "fps": (1, 60),
        "num_inference_steps": (1, 200),
        "guidance_scale": (1.0, 30.0),
        "width": (64, 2048),
        "height": (64, 2048),
        "batch_size": (1, 16),
    }
    
    # Valid string values
    VALID_PRECISIONS = ["fp16", "fp32", "bf16"]
    VALID_DEVICES = ["cpu", "cuda", "auto"]
    VALID_REFERENCE_DATASETS = ["ucf101", "kinetics600", "sky_timelapse"]
    
    def validate_benchmark_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate benchmark configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated and sanitized configuration
        """
        validated_config = {}
        
        for key, value in config.items():
            if key in self.CONFIG_RANGES:
                validated_config[key] = self._validate_numeric_range(key, value)
            elif key == "precision":
                validated_config[key] = self._validate_choice(key, value, self.VALID_PRECISIONS)
            elif key == "device":
                validated_config[key] = self._validate_choice(key, value, self.VALID_DEVICES)
            elif key == "reference_dataset":
                validated_config[key] = self._validate_choice(key, value, self.VALID_reference_datasets)
            elif key == "resolution":
                validated_config[key] = self._validate_resolution(value)
            elif key == "save_videos":
                validated_config[key] = self._validate_boolean(key, value)
            else:
                # Pass through unknown keys (with warning)
                validated_config[key] = value
                
        return validated_config
        
    def _validate_numeric_range(self, key: str, value: Any) -> Union[int, float]:
        """Validate numeric value is within allowed range."""
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise InvalidConfigError(key, value, "Must be a number")
                
        min_val, max_val = self.CONFIG_RANGES[key]
        
        if value < min_val or value > max_val:
            raise InvalidConfigError(
                key, 
                value, 
                f"Must be between {min_val} and {max_val}"
            )
            
        # Return as int if it was originally an int range
        if isinstance(min_val, int) and isinstance(max_val, int):
            return int(value)
        return float(value)
        
    def _validate_choice(self, key: str, value: Any, valid_choices: List[str]) -> str:
        """Validate value is one of allowed choices."""
        if not isinstance(value, str):
            raise InvalidConfigError(key, value, "Must be a string")
            
        if value not in valid_choices:
            raise InvalidConfigError(
                key,
                value,
                f"Must be one of: {', '.join(valid_choices)}"
            )
            
        return value
        
    def _validate_resolution(self, value: Any) -> str:
        """Validate resolution string."""
        if isinstance(value, str):
            if 'x' not in value:
                raise InvalidConfigError("resolution", value, "Must be in format 'WIDTHxHEIGHT'")
                
            try:
                width, height = map(int, value.split('x'))
            except ValueError:
                raise InvalidConfigError("resolution", value, "Must be in format 'WIDTHxHEIGHT'")
                
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            width, height = value
            value = f"{width}x{height}"
        else:
            raise InvalidConfigError("resolution", value, "Must be string 'WxH' or tuple (W, H)")
            
        # Validate dimensions
        if width < 32 or width > 2048 or height < 32 or height > 2048:
            raise InvalidConfigError(
                "resolution", 
                value, 
                "Width and height must be between 32 and 2048"
            )
            
        # Check if dimensions are multiples of 8 (common requirement)
        if width % 8 != 0 or height % 8 != 0:
            raise InvalidConfigError(
                "resolution",
                value,
                "Width and height should be multiples of 8"
            )
            
        return value
        
    def _validate_boolean(self, key: str, value: Any) -> bool:
        """Validate boolean value."""
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            lower_val = value.lower()
            if lower_val in ("true", "1", "yes", "on"):
                return True
            elif lower_val in ("false", "0", "no", "off"):
                return False
            else:
                raise InvalidConfigError(key, value, "Must be a boolean value")
        elif isinstance(value, (int, float)):
            return bool(value)
        else:
            raise InvalidConfigError(key, value, "Must be a boolean value")


class FileValidator:
    """Validates file paths and operations."""
    
    ALLOWED_EXTENSIONS = {'.json', '.yaml', '.yml', '.txt', '.csv', '.npy', '.mp4', '.avi', '.mov'}
    MAX_PATH_LENGTH = 255
    
    def validate_output_path(self, path: Union[str, Path]) -> Path:
        """Validate output file path.
        
        Args:
            path: File path to validate
            
        Returns:
            Validated Path object
        """
        if isinstance(path, str):
            path = Path(path)
            
        # Check path length
        if len(str(path)) > self.MAX_PATH_LENGTH:
            raise ValidationError(f"Path too long: {len(str(path))} > {self.MAX_PATH_LENGTH}")
            
        # Check for path traversal
        path_str = str(path)
        if '..' in path_str or path_str.startswith('/'):
            raise ValidationError("Path traversal not allowed")
            
        # Check extension
        if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise ValidationError(
                f"Invalid file extension '{path.suffix}'. "
                f"Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}"
            )
            
        # Check parent directory exists or can be created
        parent_dir = path.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise ValidationError(f"Cannot create directory {parent_dir}: {e}")
                
        return path
        
    def validate_input_path(self, path: Union[str, Path]) -> Path:
        """Validate input file path.
        
        Args:
            path: File path to validate
            
        Returns:
            Validated Path object
        """
        if isinstance(path, str):
            path = Path(path)
            
        # Check if file exists
        if not path.exists():
            raise ValidationError(f"File does not exist: {path}")
            
        # Check if it's a file (not directory)
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {path}")
            
        # Check file size (max 100MB)
        file_size = path.stat().st_size
        max_size = 100 * 1024 * 1024  # 100MB
        if file_size > max_size:
            raise ValidationError(f"File too large: {file_size} > {max_size} bytes")
            
        return path


class APIValidator:
    """Validates API requests and parameters."""
    
    def __init__(self):
        self.prompt_validator = PromptValidator()
        self.config_validator = ConfigValidator()
        
    def validate_benchmark_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate benchmark request data.
        
        Args:
            request_data: Request data dictionary
            
        Returns:
            Validated request data
        """
        validated = {}
        
        # Validate model name
        model_name = request_data.get("model_name")
        if not model_name or not isinstance(model_name, str):
            raise ValidationError("model_name is required and must be a string")
            
        if not re.match(r'^[a-zA-Z0-9_-]+$', model_name):
            raise ValidationError("model_name contains invalid characters")
            
        validated["model_name"] = model_name
        
        # Validate prompts
        prompts = request_data.get("prompts")
        if prompts is not None:
            if not isinstance(prompts, list):
                raise ValidationError("prompts must be a list")
            validated["prompts"] = self.prompt_validator.validate_batch(prompts)
        else:
            validated["prompts"] = None
            
        # Validate config
        config = request_data.get("config", {})
        if not isinstance(config, dict):
            raise ValidationError("config must be a dictionary")
            
        validated["config"] = self.config_validator.validate_benchmark_config(config)
        
        return validated
        
    def validate_pagination_params(
        self, 
        limit: Optional[int], 
        offset: Optional[int] = None,
        max_limit: int = 1000
    ) -> Tuple[int, int]:
        """Validate pagination parameters.
        
        Args:
            limit: Number of items to return
            offset: Number of items to skip
            max_limit: Maximum allowed limit
            
        Returns:
            Tuple of (validated_limit, validated_offset)
        """
        # Validate limit
        if limit is None:
            limit = 50  # Default
        elif not isinstance(limit, int) or limit < 1:
            raise ValidationError("limit must be a positive integer")
        elif limit > max_limit:
            raise ValidationError(f"limit cannot exceed {max_limit}")
            
        # Validate offset
        if offset is None:
            offset = 0
        elif not isinstance(offset, int) or offset < 0:
            raise ValidationError("offset must be a non-negative integer")
            
        return limit, offset


# Global validator instances
prompt_validator = PromptValidator()
config_validator = ConfigValidator()
file_validator = FileValidator()
api_validator = APIValidator()


# Convenience functions
def validate_prompt(prompt: str, max_length: int = 1000) -> str:
    """Validate a single prompt."""
    validator = PromptValidator(max_length=max_length)
    return validator.validate(prompt)


def validate_prompts(prompts: List[str], max_length: int = 1000) -> List[str]:
    """Validate a list of prompts."""
    validator = PromptValidator(max_length=max_length)
    return validator.validate_batch(prompts)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate benchmark configuration."""
    return config_validator.validate_benchmark_config(config)